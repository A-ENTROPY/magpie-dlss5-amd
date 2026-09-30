#include "pch.h"
#include "DlssnrAmdBackend.h"

// The reconstruction route drives the FSR3 upscaler this project already carries. Its
// implementation is compiled only when the FidelityFX SDK is present, and the build treats
// that SDK as optional, so the call sites are guarded the same way rather than assuming it.
#ifdef MP_ENABLE_FSR3_ZEROMV
#include "FSR3Upscaler.h"
#endif
#include "FrameGuidanceD3D12Interop.h"
#include "DeviceResources.h"
#include "DirectXHelper.h"
#include "Logger.h"
#include "Win32Helper.h"

// d3d12.h is not part of the precompiled header. DLSSNRFilter.cpp gets its D3D12 types
// through the NGX headers, which this backend deliberately does not include -- it has no
// business pulling NVIDIA's SDK into a path that runs on a Radeon.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <bcrypt.h>
#include <d3dcompiler.h>
// The runtime's wait loops sleep in milliseconds, and a millisecond sleep is only a
// millisecond when the system timer has been raised. See the note in Initialize.
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")

namespace Magpie {

namespace {

// ---------------------------------------------------------------------------
// The runtime's internals, for one specific build.
//
// The DLL exports nothing, so every interaction is a write to a fixed offset or a call
// to an address inside its image. All of these were read out of the working
// implementation's own source (OptiScaler PreSR-Multipass, dlssnr/amd/AmdPreSr.cpp)
// rather than inferred, and they are valid for exactly one binary -- the one the hash
// below names. A wrong offset does not fail, it jumps into nothing.
//
//   base + 0x1fe80   bool(void* ctx, const std::string* weights)   engine init
//   base + 0x12640   void(Packet*)                                 record a job
//   base + 0x9460    void(queue, n, lists)                         notify submitted
// ---------------------------------------------------------------------------
// One build is supported, and the table stays a table anyway: the offsets are derived, not
// read from source, so they need a place to live that is not a scatter of constants through
// the file with no statement of which build they describe.
//
// It is danielblnc 0.5.0 and nothing else. Earlier builds are retired rather than kept
// alongside: 0.5.0 is the first whose register-resident kernels are compiled for gfx1100 at
// all (see kOffsets050 below), it measured 82 ms -> 27 ms of network time on this card, and
// carrying the old columns would mean carrying the old kernel-substitution machinery with
// them, which is tied to their exact addresses and is now redundant.
//
// IdentifyRuntime still tells builds apart by SHA-256 rather than trusting a filename,
// because handing these offsets to a different layout does not fail -- it jumps into
// nothing. A runtime that is not 0.5.0 is refused with a message naming what it found.
struct RuntimeOffsets {
    uintptr_t init, record, notify;
    uintptr_t device, queue, initCtx, hipDevice;
    uintptr_t inlineMode, interop, enabled, useFsrInputs, useDepth, tonemap;
    uintptr_t initDone, history, wantHistory, perPassFlag;
    uintptr_t depthInverted, depthExplicit;
    uintptr_t localTone, localStructure, skinStructure, charMask, toneChannels;
    uintptr_t jobCounter, statusFlag, syncCounter, trampoline, watchdog, pendingList;
};

// The runtime calls what `trampoline` points at to hand work over. Every installation the
// reference supports points it at a no-op, because the submission is done by whoever drives
// the queue -- here, that is this backend calling Notify. Left unfilled it is a null pointer
// the runtime will call.
// danielblnc 0.5.0. The build this backend exists for.
//
// "Up to 207% performance improvement over v0.4.3" on RX 7000, per its own release notes,
// and that number is real here: measured 82.2-83.0 ms of network time at 1920x1080 on an
// RX 7900 XTX before, 27.4-27.7 ms after. It is an RDNA3 change and only an RDNA3 change --
// RX 9000 got 5% in the same release. 0.5.0's gfx1100 code object is 7,852,736 B against
// 0.4.3's 1,270,096, and of the 98 `k_reg_*` kernels 0.4.3 shipped, 96 were 52-byte
// `s_sethalt 5` stubs while 0.5.0's 111 have a median size of 84,548 B. The Fast path
// reaches RDNA3 by expanding E4M3 to F16 with packed 16-bit integer ops and running the
// matmuls on gfx11's F16 matrix units -- `k_reg_swin32<32,false>` carries 152
// `v_wmma_f32_16x16x16_f16`, 567/464/176 `v_pk_{add,mul,fma}_f16` and 528/312/248
// `v_pk_{add_u16,lshrrev_b16,min_u16}`, with no `cvt_pk_fp8` (gfx12-only) and no `s_trap`.
//
// Derived as follows, twice over, because a wrong offset here jumps into nothing rather
// than failing. The three function entries are unique 32-byte prologue hits. The 27 data
// fields are NOT a constant delta away: .data grew by exactly 0x40 bytes of virtual size
// and the shift is a staircase (0x5000/0x5008/0x5018/0x5020/0x5028/0x5038/0x503c/0x5040),
// which is what insertions look like. Two independent methods agree on all 27:
// tools/derive_050_offsets.py aligns each paired function's .data reference list and samples
// 358 old addresses, every field landing within one byte of a sample carrying a single
// undisputed delta; and the project's own tools/pair_globals.py matches on the instruction
// context around each reference, scoring 0.61-0.93 with a clear margin over the runner-up.
// Weights are unchanged: dlssnr_on_amd_weights.bin is 6bf8dc93... as it was for 0.3.0.
constexpr RuntimeOffsets kOffsets050{
    0x29870, 0x15640, 0xa000,
    0xb5c18, 0xb5c20, 0xb5c30, 0xb6ae8,
    0xb65d0, 0xb6828, 0xb69c4, 0xb69c6, 0xb69c7, 0xb69c8,
    0xb60c8, 0xb5d88, 0xb5d90, 0xb69c5,
    0xb69b8, 0xb69bc,
    0xb69d8, 0xb69dc, 0xb69e0, 0xb69e8, 0xb69ec,
    0xb6914, 0xb60ca, 0xb6604, 0xb6b88, 0xb6634, 0xb6908,
};

// How many frames may be in flight through the engine at once.
//
// Four, and the reason is the runtime's own: with inline handshaking off it reports
// "mode async (residual from an earlier frame)" and returns before its work is done, so the
// frame being recorded and the frame being read cannot be the same one. One surface means the
// second frame overwrites the first while the engine is still reading it, and the picture
// comes back black.
//
// The engine takes about three times a frame here, so with two slots the composite can never
// find one that has already retired and every frame waits out an entire engine job -- the
// frame time becomes the engine's, with its variance on top. Four covers that ratio with room,
// which is what the reference's own slot count (1-5, default 3) is for.
constexpr uint32_t kSlots = 4;

// How long a teardown will wait for the engine to publish its last job and for this queue to
// retire it. An engine job is twenty to thirty milliseconds, so this is two orders of magnitude
// of headroom; it is a bound on a hang, not a budget for normal work.
constexpr uint64_t kDrainDeadlineMs = 2000;


// The engine's substrate, for the life of the process.
//
// The runtime is loaded once and never unloaded -- ~Impl explains why -- so its engine is
// process state. Everything the engine was initialised *against* has to be process state too,
// and it was not: Magpie builds a backend per scaling session and destroys it when scaling
// stops, so the D3D12 device, the command queue and the HIP device index were all rebuilt for
// the second session and the engine re-initialised on top of the first one.
//
// That is measured, not inferred. In one Magpie process, the session that ran `engine bring-up
// 1` carried 36 seconds of frames without a fault, and the session that ran `engine bring-up 2`
// -- same process, same runtime instance -- faulted the GPU within six seconds and took the
// machine down with a kernel bugcheck. The two differ in nothing else.
//
// So the first session that needs these creates them and every later one adopts them. Only the
// size-dependent resources stay per session, and they already have their own lifetime
// (CreateSized / DestroySized).
struct EngineHost {
	winrt::com_ptr<ID3D12Device> device;
	winrt::com_ptr<ID3D12CommandQueue> queue;
	// Taken from the queue's vtable while it still belongs to the driver. Re-deriving it in a
	// later session would read it with the runtime already loaded, which is the one moment it
	// cannot be read -- so it is captured once, here, and passed on.
	void* executeOriginal = nullptr;
	// The module and the offset table it was identified as. The runtime is never unloaded, so a
	// later session has no reason to load it again: that would take the loader lock to bump a
	// refcount on a module this process already holds, and re-read and re-hash 38 MB to reach
	// the answer already here.
	HMODULE runtime = nullptr;
	const RuntimeOffsets* rva = nullptr;
	// The runtime's init has run against the objects above. There is no second time.
	bool started = false;
};

// Function-local so the order of namespace-scope initialisation cannot matter.
inline EngineHost& Engine() noexcept {
	static EngineHost host;
	return host;
}


// SHA-256 of the one build these offsets belong to. Anything else is refused rather than
// attempted: the offsets point into a different layout and the failure mode is a jump into
// nothing, not an error.
//
// It is the stock 0.5.0 image, the same bytes the installer puts in the game folder, which
// is 38,703,616 bytes and ships gfx1100/1101/1102/1200/1201 -- unpatched, deliberately.
//
// A patched build was shipped once and it does not work. The reasoning behind it looked sound
// and was wrong, so it is recorded here rather than left in a commit message:
// tools/patch_runtime_050.py neutralised the CreateThread that starts the thread installing the
// runtime's five detours (the game's ExecuteCommandLists, both CreateSwapChain overloads, both
// Presents), on the theory that the runtime driving itself and this backend driving it could not
// both hold the wheel. With the patch in place the runtime loads, stages, warms up and passes
// its inline flag check -- and then prints "apply UAV format 10" and stops. `first capture
// submitted on a direct queue` never appears, no network job is ever run, and Magpie reports
// MP-035 within seconds.
//
// The detour is not the runtime driving itself. It is how the runtime *gets a frame*: it hooks
// the game's command list to find where its capture belongs. The backend drives the network; the
// runtime's detour supplies its input. They are not competing for one wheel, and removing the
// detour removes the input.

// danielblnc 0.5.0. This is the build that matters for gfx1100: 0.4.3 and everything before
// it could only run the slow Reference path here, because the register-resident kernels it
// selected were `s_trap` stubs on RDNA3 (see kOffsets050 above). 0.4.3's own notes say as
// much -- "RX 7000 always runs Reference" -- so its table is retired rather than kept
// alongside this one.
constexpr uint8_t kRuntimeSha256_050[32] = {
	0xcd, 0xdf, 0xb0, 0x9e, 0x01, 0x93, 0x47, 0x95, 0x7b, 0xf7, 0xb9, 0x6c,
	0x95, 0xc0, 0xe9, 0x00, 0xe8, 0xd3, 0x06, 0x2d, 0xfa, 0xed, 0x69, 0x7a,
	0x8a, 0x96, 0xb0, 0xa0, 0x39, 0xae, 0xc3, 0x1a,
};

constexpr const wchar_t* kRuntimeName = L"dlssnr_amd_pass1.dll";
constexpr const wchar_t* kWeightsName = L"dlssnr_on_amd_weights.bin";
constexpr const wchar_t* kIniName = L"dlssnr_on_amd.ini";

// Every state field in the packet is this value. The reference writes it for the colour,
// the motion, the depth and the exposure alike, including at points where the resource is
// demonstrably sitting in NON_PIXEL_SHADER_RESOURCE -- so the field is a fixed token the
// engine expects to see, not a state it reconciles against. Passing the real state here
// changed nothing and cost a debug-layer diagnostic.
constexpr UINT kPacketState = 4;

// The engine leaves its output in NON_PIXEL_SHADER_RESOURCE -- the D3D12 debug layer
// named that exact state when a barrier disagreed with it. Barriers that follow the
// engine start from here.
constexpr D3D12_RESOURCE_STATES kStateShaderRead =
	D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

template <class T>
T& At(HMODULE module, uintptr_t rva) noexcept {
	return *reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(module) + rva);
}

struct Packet {
	ID3D12GraphicsCommandList* list;
	ID3D12Resource* colour;
	UINT colourState, pad14;
	ID3D12Resource* motion;
	UINT motionState, pad24;
	ID3D12Resource* depth;
	UINT depthState, pad34;
	ID3D12Resource* exposure;
	UINT exposureState;
	float scaleX, scaleY;
	// 0.3.0 grew the packet by sixteen bytes. Sending the shorter one is not a near miss:
	// the engine reads past the end of it, and the frame that comes back is black. These are
	// the fields it added, and the last two are why this route was soft -- the engine is told
	// the render extent and the sub-pixel phase of the colour it was handed, which it needs
	// to reconstruct anything the sampling did not already average away.
	uint8_t nativePre;
	uint8_t pad4d[3];
	UINT renderWidth, renderHeight;
	float jitterX, jitterY;
};
static_assert(sizeof(Packet) == 0x60, "Packet must be 0x60 bytes");
static_assert(offsetof(Packet, scaleX) == 0x44, "scaleX must sit at 0x44");
static_assert(offsetof(Packet, nativePre) == 0x4c, "nativePre must sit at 0x4c");
static_assert(offsetof(Packet, renderWidth) == 0x50, "renderWidth must sit at 0x50");
static_assert(offsetof(Packet, jitterX) == 0x58, "jitterX must sit at 0x58");

using InitFn = bool(__fastcall*)(void*, const std::string*);
using RecordFn = void(__fastcall*)(Packet*);
using NotifyFn = void(__fastcall*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
using HipSetFn = int (*)(int);

// What the runtime calls when it wants work handed to the queue. Every supported installation
// points the trampoline here: the submission is done by whoever drives the queue, and in this
// backend that is Notify, called directly. Leaving it unfilled gives the runtime a null
// pointer to call, which is not a state any working installation is in.
void __fastcall AlreadySubmitted(
	ID3D12CommandQueue*, UINT, ID3D12CommandList* const*
) noexcept {}

// The only shaders this backend owns: the two ends of a format conversion.
//
// The engine takes and returns R16G16B16A16_FLOAT, while Magpie's endpoints are usually
// eight bit -- and its input is often B8G8R8A8 where its output is R8G8B8A8. Those two
// names describe the bytes in memory, not different colours: a shader reading either one
// gets .r as red, .g as green and .b as blue, and a shader writing either one puts .r in
// the red slot. So both conversions are plain copies, and neither has any business
// reordering channels.
//
// This is worth stating plainly because an earlier version swapped red and blue on the
// way in, on the theory that the engine wanted the channels the other way round. Nothing
// swapped them back, and the picture came out looking like a colour-blindness filter --
// exactly one R/B flip, applied to every frame.
constexpr char kConvertInShader[] = R"(
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
#ifdef SRGB_IN
float3 SrgbToLinear(float3 v) {
	return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4);
}
#endif
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	float4 v = src.Load(int3(id.xy, 0));
#ifdef SRGB_IN
	v.rgb = SrgbToLinear(v.rgb);
#endif
	dst[id.xy] = v;
}
)";

constexpr char kConvertOutShader[] = R"(
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
#ifdef SRGB_IN
cbuffer Settings : register(b0) { uint shoulderMilli; };

float3 DisplayShoulder(float3 v) {
	// The target is eight bits: encoded 1.0 is 255, so any linear value above 1.0 clips to
	// flat white and the detail inside highlights is gone. A roll-off above 1.0 cannot
	// help -- the encode still pushes it past 255 -- so the top of the linear range is
	// remapped into the display range instead. Everything below the anchor is untouched;
	// above it the value compresses smoothly toward 1.0. The slope just above the anchor
	// is made to equal the slope below it, so there is no seam, and the mapping is
	// monotonic, so the brightest pixels keep their ordering. Linear 1.0 lands near 0.925
	// with the default anchor, and a value of 10 lands near 0.998, instead of everything
	// above 1.0 sharing the same flat 255.
	const float k = max(shoulderMilli, 10u) / 1000.0;
	const float s = k / (1.0 - k);   // the slope needed for a C1 join at the anchor
	v = max(v, 0);
	float3 t = (v - k) / max(v, 1e-6);      // 0 at the anchor, -> 1 at infinity
	float3 g = t / (t + (1.0 - t) / s);     // -> 0 at t=0, -> 1 at t=1
	return lerp(v, k + (1.0 - k) * g, saturate(t * 1000.0));
}

float3 LinearToSrgb(float3 v) {
	v = DisplayShoulder(v);
	return v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}
#endif
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	float4 v = src.Load(int3(id.xy, 0));
#ifdef SRGB_IN
	v.rgb = LinearToSrgb(v.rgb);
#endif
	dst[id.xy] = v;
}
)";

// The network's cost is close to linear in the pixels it is handed -- measured here at
// roughly 30 ms per megapixel plus about 15 ms per job, so 78 ms at 1080p and 266 ms at
// 4K. In inline mode the game waits for it, which puts the frame rate at 1/cost. Running
// the network on a smaller frame is therefore the only lever that moves it much, and it is
// the one the reference exposes too.
//
// The two shaders below are that lever. Neither is a plain resize: a naive downscale
// aliases and a naive upscale softens, and the whole point of the effect is the detail it
// adds. Both are taken from the reference, which arrived at the same problem.
// The bit pattern of a float, for the places that hand one to a shader as a root constant.
inline uint32_t BitsOfFloat(float v) noexcept {
	uint32_t bits = 0;
	std::memcpy(&bits, &v, 4);
	return bits;
}

// The sub-pixel phases the sampling window visits, one per frame: a Halton (2, 3) sequence,
// which spreads evenly over the unit square at every prefix length. That matters here because
// the reconstruction gets something useful out of the first frames rather than only after a
// full cycle, and there is no fixed cycle length to be caught out by.
constexpr float kPhaseX[8]{ 0.5f, 0.25f, 0.75f, 0.125f, 0.625f, 0.375f, 0.875f, 0.0625f };
constexpr float kPhaseY[8]{ 1.0f / 3, 2.0f / 3, 1.0f / 9, 4.0f / 9,
	7.0f / 9, 2.0f / 9, 5.0f / 9, 8.0f / 9 };

// Fills the shared input surface from Magpie's frame, on the D3D11 side.
//
// A plain CopyResource writes the pixels and this backend can read them back on the same
// device, so it looks like it worked -- but the other device never sees them, and the whole
// chain faithfully produces black from what it reads. Both implementations in this codebase
// that cross this boundary successfully -- the AMD optical flow provider and the older DLSSNR
// filter -- fill their shared surface through a compute shader writing a UAV rather than a
// copy, and that is the one structural difference left between this chain and theirs.
constexpr char kFillSharedShader[] = R"(
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	uint w, h;
	dst.GetDimensions(w, h);
	if (id.x >= w || id.y >= h) return;
	dst[id.xy] = src.Load(int3(id.xy, 0));
}
)";

constexpr char kDownsampleShader[] = R"(
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
cbuffer Extent : register(b0) {
	uint w; uint h; uint sourceW; uint sourceH;
	uint pointSample; float jitterX; float jitterY;
};
[numthreads(8, 8, 1)]
void main(uint3 p : SV_DispatchThreadID)
{
	if (p.x >= w || p.y >= h) return;
	float2 scale = float2(sourceW, sourceH) / float2(w, h);
	if (pointSample != 0) {
		// One sample, placed by this frame's sub-pixel phase.
		float2 at = (float2(p.xy) + 0.5 + float2(jitterX, jitterY)) * scale;
		int2 ip = clamp(int2(at), int2(0, 0), int2(int(sourceW) - 1, int(sourceH) - 1));
		dst[p.xy] = src.Load(int3(ip, 0));
		return;
	}
	// Integrate the whole source footprint. A single bilinear sample loses narrow
	// emissive lines once the model runs well below the input resolution.
	//
	// The footprint is shifted whole by this frame's sub-pixel phase, which is what gives a
	// run of frames something to accumulate: every frame carries the same information
	// otherwise, and identical samples average into themselves. Moving the window rather than
	// taking one sample from it also keeps each frame free of the aliasing a single sample
	// would hand the reconstruction to clean up first -- and that cleanup costs detail. Which
	// of the two survives is not something to reason about, so both are here.
	float2 lo = (float2(p.xy) + float2(jitterX, jitterY)) * scale;
	float2 hi = lo + scale;
	int2 first = int2(floor(lo));
	float4 sum = 0;
	float total = 0;
	[loop] for (int y = first.y; y < int(ceil(hi.y)); ++y)
	[loop] for (int x = first.x; x < int(ceil(hi.x)); ++x) {
		float2 coverage = max(0, min(hi, float2(x + 1, y + 1)) - max(lo, float2(x, y)));
		float weight = coverage.x * coverage.y;
		sum += src.Load(int3(clamp(int2(x, y), 0, int2(sourceW - 1, sourceH - 1)), 0)) * weight;
		total += weight;
	}
	dst[p.xy] = sum / max(total, 1e-6);
}
)";

// Upscales the *edit* rather than the picture: the output starts as the untouched
// full-resolution frame and gains only the bounded, confidence-weighted difference the
// network made at low resolution. That is what keeps a reduced model from costing detail.
constexpr char kResolveShader[] = R"(
Texture2D<float4> src : register(t0);
Texture2D<float4> baseline : register(t1);
Texture2D<float4> residual : register(t2);
RWTexture2D<float4> dst : register(u0);
cbuffer Extent : register(b0) { uint w; uint h; uint lowW; uint lowH; uint boundMilli; };

// Comparisons happen in a compressed domain, so a threshold means the same thing at any
// brightness. This is the upstream temporal filter's own choice for the same reason.
float3 Guide(float3 c) { return c / (1 + abs(c)); }
float MaxAbs(float3 v) { return max(abs(v.r), max(abs(v.g), abs(v.b))); }

// The edit comes from the residual the temporal pass left, which is this frame's edit when
// accumulation is off and a blend of recent frames' when it is on. It is accepted only
// where the reduced frame agrees with this pixel. Three things here come from the upstream
// anti-flicker filter, which works the same problem from the other side:
//
//   the agreement is judged over the whole bilinear footprint and by its worst sample, not
//   by one sample -- a single comparison is noisy, and a noisy weight is a weight that
//   wobbles every frame, which is what the flicker was;
//
//   the band is tight (.02 to .10) rather than wide and gentle (.15 to .75). A wide band
//   makes the weight vary continuously with tiny changes in the input, which is exactly
//   the instability; a tight one is near-binary and therefore steady;
//
//   and when the footprint does not support an answer the call reports failure and the
//   caller leaves the pixel alone, rather than publishing a half-weighted edit. The
//   upstream filter does the same (`if (mass < .05) return false`).
bool SampleEdit(int2 p, out float3 edit) {
	float3 c = Guide(src.Load(int3(p, 0)).rgb);
	float2 q = (float2(p) + .5) * float2(lowW, lowH) / float2(w, h) - .5;
	int2 a = int2(floor(q));
	float2 t = frac(q);
	float3 sum = 0;
	float mass = 0;
	float worst = 0;
	[unroll] for (int y = 0; y < 2; ++y)
	[unroll] for (int x = 0; x < 2; ++x) {
		int2 n = clamp(a + int2(x, y), 0, int2(lowW - 1, lowH - 1));
		float4 r = residual.Load(int3(n, 0));
		float4 b = baseline.Load(int3(n, 0));
		if (!all(isfinite(r.rgb)) || !all(isfinite(b.rgb))) continue;
		worst = max(worst, MaxAbs(c - Guide(b.rgb)));
		float weight = (x ? t.x : 1 - t.x) * (y ? t.y : 1 - t.y);
		sum += weight * r.rgb;
		mass += weight;
	}
	if (mass < .05) return false;
	float accept = 1 - smoothstep(.02, .10, worst);
	if (accept <= .001) return false;
	edit = (sum / mass) * accept;
	return true;
}

[numthreads(8, 8, 1)]
void main(uint3 p : SV_DispatchThreadID)
{
	if (p.x >= w || p.y >= h) return;
	float4 c = src.Load(int3(p.xy, 0));
	float3 edit;
	float3 result = c.rgb;
	// The bound is a safety net for extremes, not the mechanism; the acceptance above is
	// what decides whether an edit is applied at all.
	if (SampleEdit(int2(p.xy), edit)) {
		const float3 limit = (boundMilli / 1000.0) * max(abs(c.rgb), 1e-4);
		result = c.rgb + clamp(edit, -limit, limit);
	}
	if (!all(isfinite(result))) result = c.rgb;
	dst[p.xy] = float4(clamp(result, 0, 65504), c.a);
}
)";

// The exposure handed to the engine, in a one-pixel R32_FLOAT texture.
//
// Left to itself the engine adapts its own exposure frame by frame, and that adaptation is
// not stable. On a scene whose input mean sat between 0.490 and 0.502 the exposure it chose
// moved between 0.645 and 0.925 -- a fifth of its own value, frame to frame, with the
// picture not changing. That multiplies the whole image and reads as a lamp breathing.
//
// It is not something this backend introduced. In the working OptiScaler installation's own
// log, at the same 960x540, its exposure moved between 0.65 and 3.33 -- five times -- on
// input that never left 0.48..0.51. The reference passes no exposure and lives with it.
//
// 1.0 is the reference's own value for having nothing better: its exposure shader ends
// `dst = isfinite(e) && e > 0 ? e : 1.0`. A game frame has already been exposed by the
// game, so the network has no business re-adapting it.
constexpr float kExposureValue = 1.0f;

// The residual, carried between frames. This is upstream's `antiFlicker` mode 1, static
// accumulation, whose design document names it the one route that needs no motion vectors
// -- the only kind available here. Its shape is theirs:
//
//   the residual -- what the network changed, taken at the resolution it ran at -- is kept
//   rather than the picture, and never fed back into the network;
//
//   this frame's residual is blended toward that history, and the history is first clamped
//   into the range the current frame's neighbourhood actually spans, which is what stops a
//   correction from persisting where the picture moved on;
//
//   history is trusted only where the guide agrees with it, judged over a neighbourhood by
//   both its mean and its worst sample;
//
//   and where the history cannot be trusted the frame's own residual is used unchanged,
//   rather than a blend of the two.
//
// The blend weight is not here: it comes from the capture interval, so the accumulation is
// a duration in real time rather than a frame count.
constexpr char kTemporalShader[] = R"(
Texture2D<float4> edited : register(t0);
Texture2D<float4> baseline : register(t1);
Texture2D<float4> historyResidual : register(t2);
Texture2D<float4> historyGuide : register(t3);
Texture2D<float2> motionField : register(t4);
RWTexture2D<float4> nextResidual : register(u0);
RWTexture2D<float4> nextGuide : register(u1);
cbuffer Settings : register(b0) {
	// weightMilli, not a float. The constant buffer slot is filled from a uint32 holding
	// thousandths, so declaring it float here reinterpreted the bit pattern instead of
	// converting it: 1000 became the denormal 1.4e-42, and since the only thing the weight
	// does is scale the history term of the blend below, the accumulation silently did
	// nothing -- every anti-flicker mode behaved the same, and the reprojected history the
	// optical flow exists to supply was multiplied away. Declared as the integer it is.
	uint w; uint h; uint weightMilli; uint historyValid;
	uint useMotion; uint motionScaleXMilli; uint motionScaleYMilli;
	// The residual controls, in the order the D3D11 route's ResampleConstants uses, so a slider
	// means the same thing on both paths.
	float residualMultiplier; float residualSaturation; float residualLightness;
	float shadowStructure; float reflectionGlow;
};

float3 Guide(float3 c) { return c / (1 + abs(c)); }
float MaxAbs(float3 v) { return max(abs(v.r), max(abs(v.g), abs(v.b))); }

// ---- the residual controls ----
//
// Ported from the same-named pass in the D3D11 route so a slider means the same thing on both
// paths, and applied where that route applies it: once per network-resolution pixel, to the raw
// difference between what the engine produced and what it was given, before the history blend
// and before any reprojection. The order is the point -- these controls decide which parts of
// the edit survive at all, and the temporal blend then decides how long they persist.
//
// The multiplier scales the whole edit. The other four are a refinement that only engages once
// one of them leaves 1.0: the edit is classified by the sign of its luminance change, so a
// darkening edit answers to the shadow control and a brightening one to the reflection control,
// and the result's saturation and lightness are then pulled toward or away from the original's.
float3 ToLinear(float3 c) {
	return float3(
		c.r <= 0.04045 ? c.r / 12.92 : pow(max(c.r + 0.055, 0.0) / 1.055, 2.4),
		c.g <= 0.04045 ? c.g / 12.92 : pow(max(c.g + 0.055, 0.0) / 1.055, 2.4),
		c.b <= 0.04045 ? c.b / 12.92 : pow(max(c.b + 0.055, 0.0) / 1.055, 2.4));
}

float3 RGBToHSL(float3 color) {
	float maximum = max(color.r, max(color.g, color.b));
	float minimum = min(color.r, min(color.g, color.b));
	float delta = maximum - minimum;
	float lightness = (maximum + minimum) * 0.5;
	if (delta <= 1e-6) return float3(0.0, 0.0, lightness);
	float hue = 0.0;
	if (maximum == color.r) {
		hue = (color.g - color.b) / delta;
		if (hue < 0.0) hue += 6.0;
	} else if (maximum == color.g) {
		hue = (color.b - color.r) / delta + 2.0;
	} else {
		hue = (color.r - color.g) / delta + 4.0;
	}
	float saturation = delta / max(1.0 - abs(2.0 * lightness - 1.0), 1e-6);
	return float3(hue / 6.0, saturate(saturation), saturate(lightness));
}

float HueToRGB(float p, float q, float hue) {
	hue = frac(hue);
	if (hue < 1.0 / 6.0) return p + (q - p) * 6.0 * hue;
	if (hue < 1.0 / 2.0) return q;
	if (hue < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - hue) * 6.0;
	return p;
}

float3 HSLToRGB(float3 hsl) {
	if (hsl.y <= 1e-6) return float3(hsl.z, hsl.z, hsl.z);
	float q = hsl.z < 0.5 ? hsl.z * (1.0 + hsl.y) : hsl.z + hsl.y - hsl.z * hsl.y;
	float p = 2.0 * hsl.z - q;
	return saturate(float3(HueToRGB(p, q, hsl.x + 1.0 / 3.0),
		HueToRGB(p, q, hsl.x), HueToRGB(p, q, hsl.x - 1.0 / 3.0)));
}

float3 ApplyResidualControls(float3 original, float3 residual) {
	residual *= residualMultiplier;
	if (all(residual == 0.0)) return original;
	float4 fine = float4(residualSaturation, residualLightness, shadowStructure,
		reflectionGlow);
	float3 output = saturate(original + residual);
	[branch]
	if (any(abs(fine - 1.0) >= 1e-6)) {
		float deltaY = dot(ToLinear(output) - ToLinear(original),
			float3(0.2126, 0.7152, 0.0722));
		float directional = deltaY < 0.0 ? shadowStructure :
			(deltaY > 0.0 ? reflectionGlow : 1.0);
		float3 candidate = saturate(original + residual * directional);
		[branch]
		if (abs(residualSaturation - 1.0) >= 1e-6 ||
			abs(residualLightness - 1.0) >= 1e-6) {
			float3 originalHSL = RGBToHSL(original);
			float3 candidateHSL = RGBToHSL(candidate);
			candidateHSL.y = saturate(originalHSL.y +
				(candidateHSL.y - originalHSL.y) * residualSaturation);
			candidateHSL.z = saturate(originalHSL.z +
				(candidateHSL.z - originalHSL.z) * residualLightness);
			candidate = HSLToRGB(candidateHSL);
		}
		output = candidate;
	}
	return output;
}


[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	int2 p = int2(id.xy);
	if (p.x >= int(w) || p.y >= int(h)) return;
	int2 last = int2(w, h) - 1;

	float4 raw = edited.Load(int3(p, 0));
	float4 base = baseline.Load(int3(p, 0));
	bool finiteInput = all(isfinite(raw.rgb)) && all(isfinite(base.rgb));
	float3 guide = finiteInput ? Guide(base.rgb) : 0;
	nextGuide[p] = float4(guide, finiteInput ? 1 : 0);
	if (!finiteInput) { nextResidual[p] = 0; return; }

	// The engine's edit, after the controls. Everything downstream -- the history blend and
	// the resolve -- sees the controlled residual, so one slider position means the same
	// thing all the way through.
	float3 current = ApplyResidualControls(base.rgb, raw.rgb - base.rgb) - base.rgb;
	float3 result = current;

	if (historyValid != 0 && weightMilli > 0) {
		// The range this frame's own neighbourhood spans; a carried edit may only persist as
		// far as the picture allows it to.
		float3 lo = current, hi = current;
		[unroll] for (int y = -1; y <= 1; ++y)
		[unroll] for (int x = -1; x <= 1; ++x) {
			int2 n = clamp(p + int2(x, y), int2(0, 0), last);
			float4 neighbour = edited.Load(int3(n, 0));
			float4 neighbourBase = baseline.Load(int3(n, 0));
			if (all(isfinite(neighbour.rgb)) && all(isfinite(neighbourBase.rgb))) {
				float3 residualN = neighbour.rgb - neighbourBase.rgb;
				lo = min(lo, residualN);
				hi = max(hi, residualN);
			}
		}

		float3 old = 0;
		float trust = 0;
		bool haveHistory = false;
		if (useMotion != 0) {
			// Where was this pixel last frame, by the flow. Every tap is validated against
			// the guide it was measured against before it counts: a tap whose neighbourhood
			// has moved on must not leak into the blend. Upstream's own history gather.
			float2 mv = motionField.Load(int3(p, 0));
			float2 prev = float2(p) + (all(isfinite(mv)) ? mv : 0) *
				float2(motionScaleXMilli, motionScaleYMilli) / 1000.0;
			int2 baseP = int2(floor(prev));
			float2 f = frac(prev);
			[unroll] for (int y = 0; y < 2; ++y)
			[unroll] for (int x = 0; x < 2; ++x) {
				int2 n = clamp(baseP + int2(x, y), int2(0, 0), last);
				float4 hg = historyGuide.Load(int3(n, 0));
				float4 hr = historyResidual.Load(int3(n, 0));
				if (!all(isfinite(hg.rgb)) || !all(isfinite(hr.rgb)) ||
					hg.a < .999 || hr.a < .999) continue;
				float wgt = (x ? f.x : 1 - f.x) * (y ? f.y : 1 - f.y);
				float accept = 1 - smoothstep(.025, .10, MaxAbs(guide - hg.rgb));
				old += wgt * accept * hr.rgb;
				trust += wgt * accept;
			}
			haveHistory = trust >= .25;
			if (haveHistory) {
				old /= trust;
				trust = min(trust, 1);
			}
		} else {
			// No motion: the history is read where it is and trusted only where the guide
			// still agrees, judged by the mean and the worst of the neighbourhood.
			float error = 0, worst = 0;
			bool valid = true;
			[unroll] for (int y = -1; y <= 1; ++y)
			[unroll] for (int x = -1; x <= 1; ++x) {
				int2 n = clamp(p + int2(x, y), int2(0, 0), last);
				float4 hg = historyGuide.Load(int3(n, 0));
				if (!all(isfinite(hg.rgb)) || hg.a < .999) { valid = false; continue; }
				float e = MaxAbs(guide - hg.rgb);
				error += e / 9.0;
				worst = max(worst, e);
			}
			float4 h = historyResidual.Load(int3(p, 0));
			if (valid && all(isfinite(h.rgb)) && h.a >= .999) {
				trust = (1 - smoothstep(.008, .04, error)) *
					(1 - smoothstep(.025, .10, worst));
				if (trust > 0) {
					old = h.rgb;
					haveHistory = true;
				}
			}
		}

		if (haveHistory) {
			float3 margin = .02 + trust * abs(old);
			float3 safe = clamp(old, lo - margin, hi + margin);
			result = lerp(current, safe, weightMilli / 1000.0 * trust);
		}
	}

	nextResidual[p] = float4(clamp(result, -65504, 65504), 1);
}
)";


// The optical-flow resample. The vectors themselves are left unchanged; the packet's scale
// fields convert their pixel scale, which is the convention the reference uses for exactly
// this hand-off. The guide Magpie produces is in source pixels, current-to-previous.
constexpr char kMotionShader[] = R"(
Texture2D<float2> src : register(t0);
RWTexture2D<float2> dst : register(u0);
cbuffer Extent : register(b0) { uint w; uint h; uint sourceW; uint sourceH; };
[numthreads(8, 8, 1)]
void main(uint3 p : SV_DispatchThreadID)
{
	if (p.x >= w || p.y >= h) return;
	if (sourceW == 0 || sourceH == 0) return;
	uint2 q = min(uint2((float2(p.xy) + 0.5) * float2(sourceW, sourceH) / float2(w, h)),
		uint2(sourceW - 1, sourceH - 1));
	dst[p.xy] = src.Load(int3(q, 0));
}
)";

std::filesystem::path ExeDirectory() noexcept {
	return Win32Helper::GetExePath().parent_path();
}



// Fills `out` with the file's SHA-256, or returns false when it cannot be read.
bool RuntimeDigest(const std::filesystem::path& file, uint8_t out[32]) noexcept {
	// Two small RAII wrappers rather than wil::unique_bcrypt_*: the WIL in use here does
	// not provide those, and the lifetimes are one line each anyway.
	struct AlgGuard {
		BCRYPT_ALG_HANDLE handle = nullptr;
		~AlgGuard() { if (handle) BCryptCloseAlgorithmProvider(handle, 0); }
	} alg;
	struct HashGuard {
		BCRYPT_HASH_HANDLE handle = nullptr;
		~HashGuard() { if (handle) BCryptDestroyHash(handle); }
	} hash;

	wil::unique_hfile file_handle(CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ,
		nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
	if (!file_handle) {
		return false;
	}
	if (BCryptOpenAlgorithmProvider(&alg.handle, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) {
		return false;
	}

	DWORD objectLength = 0, written = 0;
	if (BCryptGetProperty(alg.handle, BCRYPT_OBJECT_LENGTH,
		reinterpret_cast<PUCHAR>(&objectLength), sizeof(objectLength), &written, 0) < 0) {
		return false;
	}
	std::vector<uint8_t> object(objectLength);

	if (BCryptCreateHash(alg.handle, &hash.handle, object.data(), objectLength,
		nullptr, 0, 0) < 0) {
		return false;
	}

	std::vector<uint8_t> buffer(64 * 1024);
	DWORD read = 0;
	do {
		if (!ReadFile(file_handle.get(), buffer.data(),
			static_cast<DWORD>(buffer.size()), &read, nullptr)) {
			return false;
		}
		if (read && BCryptHashData(hash.handle, buffer.data(), read, 0) < 0) {
			return false;
		}
	} while (read);

	uint8_t digest[32]{};
	if (BCryptFinishHash(hash.handle, digest, sizeof(digest), 0) < 0) {
		return false;
	}
	std::memcpy(out, digest, sizeof(digest));
	return true;
}

// Returns the offset table for whichever build `file` is, or null when it is neither. The two
// are told apart here rather than at the call site because this is the only place the whole
// file is read anyway.
const RuntimeOffsets* IdentifyRuntime(const std::filesystem::path& file) noexcept {
	uint8_t digest[32]{};
	if (!RuntimeDigest(file, digest)) {
		return nullptr;
	}
	if (std::memcmp(digest, kRuntimeSha256_050, sizeof(digest)) == 0) {
		return &kOffsets050;
	}
	// Name what was found instead, because "it did not work" is not actionable and the
	// usual cause is a runtime from a different release sitting in the folder.
	std::string hex;
	hex.reserve(64);
	for (uint8_t b : digest) {
		hex += "0123456789abcdef"[b >> 4];
		hex += "0123456789abcdef"[b & 0xf];
	}
	Logger::Get().Error(
		"DLSSNR AMD: dlssnr_amd_pass1.dll is not the 0.5.0 build this backend carries offsets "
		"for (sha256 " + hex + "). It is refused rather than guessed at -- these offsets into a "
		"different layout jump into nothing. Earlier releases are retired: 0.5.0 is the first "
		"whose register-resident kernels are compiled for gfx1100 at all, and it is about three "
		"times faster here than anything before it.");
	return nullptr;
}

// The runtime is called through raw addresses into someone else's image, so a fault is
// a possible outcome rather than a bug report -- a wrong offset jumps into nothing. SEH
// is the only way to survive that, and MSVC refuses __try in any function needing object
// unwinding, which is why these are free functions over raw pointers.
bool CallInit(InitFn fn, void* ctx, const std::string* weights) noexcept {
	__try {
		return fn(ctx, weights);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		return false;
	}
}

bool CallRecord(RecordFn fn, Packet* packet) noexcept {
	__try {
		fn(packet);
		return true;
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		return false;
	}
}

bool CallNotify(NotifyFn fn, ID3D12CommandQueue* queue, UINT count,
	ID3D12CommandList* const* lists) noexcept {
	__try {
		fn(queue, count, lists);
		return true;
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		return false;
	}
}

// The device's own account of what went wrong, kept only when the settings file asks for the
// debug layer. This route has been failing at Close() with the device already gone, and Close
// reports that fact rather than its cause; these messages carry the cause.
winrt::com_ptr<ID3D12InfoQueue> g_deviceMessages;

void DumpDeviceMessages(const char* where) noexcept {
	if (!g_deviceMessages) {
		return;
	}
	const UINT64 count = g_deviceMessages->GetNumStoredMessages();
	for (UINT64 i = 0; i < count; ++i) {
		SIZE_T length = 0;
		if (FAILED(g_deviceMessages->GetMessage(i, nullptr, &length)) || !length) {
			continue;
		}
		// The message is pointer-aligned and a byte buffer is not, so the storage is
		// over-aligned rather than assumed to be. One message runs to a few hundred bytes.
		alignas(D3D12_MESSAGE) uint8_t storage[8192];
		if (length > sizeof(storage)) {
			continue;
		}
		D3D12_MESSAGE* message = reinterpret_cast<D3D12_MESSAGE*>(storage);
		if (FAILED(g_deviceMessages->GetMessage(i, message, &length))) {
			continue;
		}
		const size_t text = message->DescriptionByteLength > 0
			? message->DescriptionByteLength - 1 : 0;
		Logger::Get().Warn(fmt::format("DLSSNR AMD device ({}): [{}] {}",
			where, static_cast<int>(message->Severity),
			std::string_view(message->pDescription ? message->pDescription : "", text)));
	}
	g_deviceMessages->ClearStoredMessages();
}

ID3D12Device* CreateDeviceOnAdapter(ID3D11Device* device11) noexcept {
	// Each of the three ways this can fail says which it was, and the third says which adapter
	// it was trying. They used to share one message further up, which made a session that
	// landed on the wrong GPU and a device the driver refused look identical in the log.
	// This machine has two adapters, so that distinction is worth keeping.
	winrt::com_ptr<IDXGIDevice> dxgiDevice;
	HRESULT hr = device11->QueryInterface(IID_PPV_ARGS(dxgiDevice.put()));
	if (FAILED(hr)) {
		Logger::Get().Error(fmt::format(
			"DLSSNR AMD: the renderer's D3D11 device is not an IDXGIDevice (0x{:08x})",
			static_cast<uint32_t>(hr)));
		return nullptr;
	}
	winrt::com_ptr<IDXGIAdapter> adapter;
	hr = dxgiDevice->GetAdapter(adapter.put());
	if (FAILED(hr)) {
		Logger::Get().Error(fmt::format(
			"DLSSNR AMD: the renderer's device reports no adapter (0x{:08x})",
			static_cast<uint32_t>(hr)));
		return nullptr;
	}

	DXGI_ADAPTER_DESC desc{};
	const bool named = SUCCEEDED(adapter->GetDesc(&desc));
	// Which adapter this session landed on, said every time rather than only on failure: this
	// machine has two, and a session on the wrong one cannot start the engine at all.
	if (named) {
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD: render adapter vendor 0x{:04x} device 0x{:04x}",
			desc.VendorId, desc.DeviceId));
	}
	// Asked for by the settings file, not by the build, so an ordinary session is not paying
	// for it:   [DlssNrOnAmd]   DebugLayer=1
	if (GetPrivateProfileIntW(L"DlssNrOnAmd", L"DebugLayer", 0,
			(ExeDirectory() / kIniName).c_str()) != 0) {
		winrt::com_ptr<ID3D12Debug> debug;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(debug.put())))) {
			debug->EnableDebugLayer();
			// The CPU-side layer cannot see a barrier the command list records and the GPU
			// mishandles, which is exactly the shape of this route's failure.
			winrt::com_ptr<ID3D12Debug1> debug1;
			if (SUCCEEDED(debug->QueryInterface(IID_PPV_ARGS(debug1.put())))) {
				debug1->SetEnableGPUBasedValidation(TRUE);
			}
			Logger::Get().Info("DLSSNR AMD: the debug layer is on");
		} else {
			Logger::Get().Warn("DLSSNR AMD: the debug layer was asked for but is not installed");
		}
	}
	// DRED, before the device exists.
	//
	// The runtime asks for DRED itself and its log says so -- "DRED enabled (page-fault
	// reporting; breadcrumbs off)" -- but it does that during its initialisation, long after
	// this device was created, and these settings are only read at creation time. Forcing them
	// on here is what lets that reporting name the unfinished command list, which is the one
	// piece of evidence every other reading in this file cannot supply: something in the
	// engine's initialisation stops this process's D3D12 from executing submitted work, and
	// DRED is the only instrument in the process that can say which submission it was.
	{
		winrt::com_ptr<ID3D12DeviceRemovedExtendedDataSettings> dred;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(dred.put())))) {
			dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			Logger::Get().Info(
				"DLSSNR AMD: DRED breadcrumbs and page faults forced on, before the device");
		} else {
			Logger::Get().Warn("DLSSNR AMD: the DRED settings are unavailable");
		}
	}
	winrt::com_ptr<ID3D12Device> device12;
	hr = D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_12_0,
		IID_PPV_ARGS(device12.put()));
	if (FAILED(hr)) {
		Logger::Get().Error(fmt::format(
			"DLSSNR AMD: no D3D12 device at feature level 12_0 on the adapter the renderer "
			"is using -- vendor 0x{:04x} device 0x{:04x}{} -- 0x{:08x}",
			named ? desc.VendorId : 0u, named ? desc.DeviceId : 0u,
			named && desc.VendorId == 0x8086 ? " (Intel; the engine needs the AMD one)" : "",
			static_cast<uint32_t>(hr)));
		return nullptr;
	}
	// Nothing comes of it being absent: the layer writes here only when it is on.
	(void)device12->QueryInterface(IID_PPV_ARGS(g_deviceMessages.put()));
	return device12.detach();
}

// Ownership of the D3D11 side stays outside; both com_ptrs are overwritten.
bool CreateSharedTexture(
	ID3D11Device5* device11,
	ID3D12Device* device12,
	uint32_t width,
	uint32_t height,
	DXGI_FORMAT format,
	bool allowUav,
	winrt::com_ptr<ID3D11Texture2D>& out11,
	winrt::com_ptr<ID3D12Resource>& out12
) noexcept {
	D3D11_TEXTURE2D_DESC desc{};
	desc.Width = width;
	desc.Height = height;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Format = format;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE |
		(allowUav ? D3D11_BIND_UNORDERED_ACCESS : 0);
	desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

	if (FAILED(device11->CreateTexture2D(&desc, nullptr, out11.put()))) {
		return false;
	}

	winrt::com_ptr<IDXGIResource1> dxgiResource;
	if (FAILED(out11->QueryInterface(IID_PPV_ARGS(dxgiResource.put())))) {
		return false;
	}
	HANDLE raw = nullptr;
	if (FAILED(dxgiResource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &raw))) {
		return false;
	}
	wil::unique_handle handle(raw);

	return SUCCEEDED(device12->OpenSharedHandle(handle.get(), IID_PPV_ARGS(out12.put())));
}

// Three flags here are not interchangeable: ALLOW_UNORDERED_ACCESS is required for the
// engine to write, ALLOW_RENDER_TARGET is required because the packet declares
// RENDER_TARGET and the engine transitions from it, and the initial state must be
// NON_PIXEL_SHADER_RESOURCE rather than COMMON. Getting any of them wrong does not fail
// -- the engine records, reports healthy, and produces nothing.
bool CreateEngineTexture(
	ID3D12Device* device,
	uint32_t width,
	uint32_t height,
	DXGI_FORMAT format,
	winrt::com_ptr<ID3D12Resource>& out
) noexcept {
	D3D12_HEAP_PROPERTIES heap{};
	heap.Type = D3D12_HEAP_TYPE_DEFAULT;

	D3D12_RESOURCE_DESC desc{};
	desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	desc.Width = width;
	desc.Height = height;
	desc.DepthOrArraySize = 1;
	desc.MipLevels = 1;
	desc.Format = format;
	desc.SampleDesc.Count = 1;
	desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
		D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

	return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
		&desc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
		IID_PPV_ARGS(out.put())));
}

void Barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
	D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) noexcept {
	D3D12_RESOURCE_BARRIER b{};
	b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	b.Transition.pResource = resource;
	b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	b.Transition.StateBefore = before;
	b.Transition.StateAfter = after;
	list->ResourceBarrier(1, &b);
}

}


// Diagnostic only: enough half-to-float to average a texture's contents.
static float HalfToFloat(uint16_t h) noexcept {
	const uint32_t sign = uint32_t(h & 0x8000u) << 16;
	uint32_t e = (h >> 10) & 0x1Fu;
	uint32_t m = h & 0x3FFu;
	uint32_t x;
	if (e == 0) {
		if (m == 0) { x = sign; }
		else {
			e = 127 - 15 + 1;
			while ((m & 0x400u) == 0) { m <<= 1; --e; }
			m &= 0x3FFu;
			x = sign | (e << 23) | (m << 13);
		}
	} else if (e == 31) {
		x = sign | 0x7F800000u | (m << 13);
	} else {
		x = sign | ((e - 15 + 127) << 23) | (m << 13);
	}
	float f;
	std::memcpy(&f, &x, 4);
	return f;
}

struct DlssnrAmdBackend::Impl {
	ID3D11Device5* device11 = nullptr;
	ID3D11DeviceContext4* context11 = nullptr;

	winrt::com_ptr<ID3D12Device> device12;
	winrt::com_ptr<ID3D12CommandQueue> queue;
	winrt::com_ptr<ID3D12CommandAllocator> allocator;
	winrt::com_ptr<ID3D12GraphicsCommandList> list;
	// A second pair, for the compositing pass alone. The first pair is submitted to the engine
	// and is still in flight while this frame's picture is built -- resetting its allocator
	// before the GPU has finished with it fails, which is what turned the whole effect off.
	// Two submissions, two allocators, and neither waits on the other.
	winrt::com_ptr<ID3D12CommandAllocator> outAllocator;
	winrt::com_ptr<ID3D12GraphicsCommandList> outList;
	// A third pair, for readbacks alone. Measure used the main pair, which is being recorded
	// into at the moment it is called -- it reset an allocator the GPU still had, and every
	// figure it produced was zero regardless of what the surface held. A control surface known
	// to contain 1.0 read back as 0.0000, which is how that was found.
	winrt::com_ptr<ID3D12CommandAllocator> crossTestAllocator;
	winrt::com_ptr<ID3D12GraphicsCommandList> crossTestList;
	winrt::com_ptr<ID3D12Fence> crossTestFence;
	HANDLE crossTestEvent = nullptr;
	uint64_t crossTestFenceValue = 0;
	winrt::com_ptr<ID3D12CommandAllocator> measureAllocator;
	winrt::com_ptr<ID3D12GraphicsCommandList> measureList;
	// The readback's own queue, and the reason it exists.
	//
	// Measure used to submit its probe copy on the engine's queue -- the one the runtime hooks.
	// A submission the hook does not recognise is not guaranteed to run, and the failure is not
	// an error: the readback buffer simply keeps the previous call's contents. Every figure then
	// returns the last texture's value, two surfaces of the same size read identically whatever
	// they hold, and a chain that produced nothing is indistinguishable from one that works.
	// A queue of this backend's own takes the probe out of that path entirely.
	winrt::com_ptr<ID3D12CommandQueue> probeQueue;
	winrt::com_ptr<ID3D12Fence> probeFence;
	uint64_t probeFenceValue = 0;

	// The queue's own ExecuteCommandLists, taken from its vtable before the engine initialises.
	//
	// Everything this backend submits stops executing once the runtime has initialised. A clear
	// followed by a copy into a readback buffer reads back zero, on this queue and on one made
	// afterwards alike, while the fence that follows still reaches its value -- the signature of
	// an interception that consumes the call without forwarding it, since Signal is a different
	// slot on the same vtable and still runs. The engine keeps reporting jobs throughout because
	// its work runs on HIP and never goes through these queues at all.
	//
	// It also explains the engine's own inline complaint, "a store from the game's queue was NOT
	// seen by the GPU wait": that store is recorded into this list, gets submitted, and never
	// reaches the GPU.
	//
	// Taking the entry first and calling it directly is what the reference does for the same
	// reason -- its bridge saves this same vtable slot before hooking the queue -- and index 10
	// is where ExecuteCommandLists sits: past IUnknown, ID3D12Object, ID3D12DeviceChild, and the
	// pageable base that adds no methods of its own.
	using ExecuteFn = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT,
		ID3D12CommandList* const*);
	ExecuteFn executeOriginal = nullptr;
	void SubmitTo(ID3D12CommandQueue* q, UINT n, ID3D12CommandList* const* lists) const noexcept {
		if (executeOriginal) {
			executeOriginal(q, n, lists);
		} else {
			q->ExecuteCommandLists(n, lists);
		}
	}
	winrt::com_ptr<ID3D12Fence> fence;
	// The handshake the two devices need before anything crosses. Each keeps its own view of
	// one fence, and the D3D11 side waits on it where the work was submitted, so the read that
	// follows is ordered after the write rather than merely later than it.
	//
	// Waiting here on the CPU was not enough, and the shape of that failure is what took a
	// session to see: the queue had finished, the data was written, the read still happened
	// against a surface the other device had not been told about, and the driver removed the
	// device. It went unnoticed for a while because turning the debug layer on slows this side
	// down enough that the two never actually overlapped.
	winrt::com_ptr<ID3D11Fence> crossFence11;
	winrt::com_ptr<ID3D12Fence> crossFence12;
	uint64_t crossValue = 0;
	// How much of the encode list gets recorded, from the settings file. Four is the whole of
	// it; each smaller number stops one call earlier, which is how this route was narrowed
	// down after every reading of the source failed to explain it:
	//   [DlssNrOnAmd]   EncodeSteps=1
	// 5 is the whole of it: 1 barrier, 2 bind, 3 constants, 4 dispatch, 5 the copy across.
	int encodeSteps = 5;
	// This frame's sub-pixel phase, and where the sequence is. Only the reconstruction route
	// advances it: it is the only consumer that accumulates across frames.
	float frameJitterX = 0.0f;
	float frameJitterY = 0.0f;
	uint32_t jitterPhase = 0;
	// Off only to measure what it is worth; see the reading printed on the first frame.
	bool jitterEnabled = true;
	// Whether the phase moves a single sample or the whole footprint. Two shapes for one idea,
	// kept apart because sampled detail and averaged detail fail differently and the figures
	// decide which one this route wants.
	bool pointSample = true;

	struct EventGuard {
		HANDLE handle = nullptr;
		~EventGuard() { if (handle) CloseHandle(handle); }
		bool valid() const noexcept { return handle != nullptr; }
		HANDLE get() const noexcept { return handle; }
		void reset(HANDLE h) noexcept { if (handle) CloseHandle(handle); handle = h; }
	} fenceEvent;
	uint64_t fenceValue = 0;

	// Magpie's frame, on both devices.
	winrt::com_ptr<ID3D11Texture2D> sharedIn11;
	winrt::com_ptr<ID3D12Resource> sharedIn12;
	winrt::com_ptr<ID3D11Texture2D> sharedOut11;
	winrt::com_ptr<ID3D12Resource> sharedOut12;

	DXGI_FORMAT inputFormat = DXGI_FORMAT_UNKNOWN;
	DXGI_FORMAT outputFormat = DXGI_FORMAT_UNKNOWN;
	bool inputIsFp16 = false;
	bool outputIsFp16 = false;

	// The engine carries temporal history from frame to frame, and nothing used to tell it
	// when that history stopped being valid. The working implementation resets on a resize,
	// a change of settings or guides, an engine timeout, and any gap longer than a quarter
	// of a second -- that last one because a loading screen or a paused window leaves
	// history describing a scene that is no longer on screen. Blending such history forward
	// is what makes the result look like the filter has been applied several times over.
	uint64_t lastSubmitTick = 0;
	bool resetHistory = true;

	// The surface being recorded into this frame, and the job number that was submitted for
	// each one. The picture is built from the other slot -- the newest one the engine has
	// actually finished -- and waiting on that job is what makes "finished" mean something.
	uint32_t slot = 0;
	// The slot the picture was built from, so the temporal pass reprojects against the same
	// generation the resolve is reading rather than against a fixed neighbour.
	uint32_t pictureSlot = 0;
	// The generation the temporal history was last advanced for. Sentinel, so the first frame
	// always runs it.
	uint32_t temporalSourceSlot = 0xFFFFFFFFu;
	uint32_t slotJob[kSlots]{};
	// The fence value this queue reaches when the frame in that slot is finished on the GPU.
	//
	// This, and not the runtime's own counters, is what says a slot (or one of its passes) has
	// actually retired. The runtime publishes progress of its own -- jobId/jobDone -- but with
	// the asynchronous handshake the done counter is not advanced at all, which is what left
	// this waiting on a number that never moved.
	uint64_t slotFence[kSlots]{};
	// Where the queue is when the last composited frame finished. The output list has its own
	// submission, so it needs its own retirement point: resetting its allocator while the GPU
	// still holds the previous one fails, and that failure turned the effect off entirely.
	uint64_t outFence = 0;

	// The reference's per-slot state, verbatim in meaning: a slot is recorded, then submitted,
	// and only then eligible to retire. Between Record and the submission the slot is neither
	// free nor usable -- that is the state which stops another frame being recorded into it,
	// and the state that tells retirement not to look at it yet.
	struct SlotState {
		bool recorded = false;
		bool submitted = false;
		uint64_t recordedAt = 0;
		uint64_t submittedAt = 0;
		void Record(uint64_t now) noexcept {
			*this = {};
			recorded = true;
			recordedAt = now;
		}
		void Submit(uint64_t now) noexcept {
			submitted = true;
			submittedAt = now;
		}
		// Recorded but not yet submitted: this slot must not be recorded into again.
		bool BlocksRecord() const noexcept { return recorded && !submitted; }
		// D3D12 reports UINT64_MAX on device removal, not successful completion.
		bool CanRetire(bool nativeDone, uint64_t completed, uint64_t target) const noexcept {
			return recorded && submitted && target != 0 && nativeDone &&
				completed != (std::numeric_limits<uint64_t>::max)() && completed >= target;
		}
	};
	std::array<SlotState, kSlots> slotState{};
	// One ordered timeline across every slot, as the reference keeps it: each submission takes
	// the next number and signals the fence with it.
	uint64_t serial = 0;
	// Which adapter each device actually landed on. Shared handles are created on one device
	// and opened on the other, and that succeeds even when the two are not the same physical
	// adapter -- the failure then is silent: every read comes back as the surface's initial
	// contents, which is zero. Nothing else in this chain explains both directions being empty
	// at once while every handle and fence reports success.
	LUID luid11{};
	LUID luid12{};
	// The adapter the renderer's device is on, kept for diagnostics that need to build a
	// second device against the same hardware.
	winrt::com_ptr<IDXGIAdapter> device11Adapter;

	// The handshake across the device boundary, for the direction this backend actually uses:
	// Magpie's frame is copied into a shared texture on D3D11 and read out of it on D3D12.
	// Both devices touch the same memory, and being on the same machine does not order them --
	// a flush on one device is a submission, not a completion. Without this the far side reads
	// whatever was there before the copy, which at startup is nothing at all, and every stage
	// downstream faithfully produces black from it.
	// The D3D11 side of the crossing: a compute shader that fills the shared input surface,
	// and the views it needs. See kFillSharedShader for why a copy is not enough.
	winrt::com_ptr<ID3D11ComputeShader> fillSharedShader;
	winrt::com_ptr<ID3D11UnorderedAccessView> sharedInUav;
	winrt::com_ptr<ID3D11ShaderResourceView> frameSrv;
	ID3D11Texture2D* frameSrvSource = nullptr;

	winrt::com_ptr<ID3D11Fence> inFence11;
	winrt::com_ptr<ID3D12Fence> inFence12;
	uint64_t inFenceValue = 0;
	// The other direction, and a fence of its own. A fence carries one value, so a second
	// direction signalling the same object lets whichever side runs ahead satisfy the other's
	// wait the moment it does -- which is the same as not waiting at all, and is why the
	// crossing read as present in the code and absent in the picture.
	winrt::com_ptr<ID3D11Fence> outFence11;
	winrt::com_ptr<ID3D12Fence> outFence12;
	uint64_t outFenceValue = 0;
	// The picture is produced on D3D12 and read on D3D11, and this queue finishing says nothing
	// about the other device being able to see it: waiting on our own fence proves our work is
	// done, not that it is visible elsewhere.
	winrt::com_ptr<ID3D11DeviceContext4> context11x;

	// The engine's surfaces, and there are two of each.
	//
	// The engine works in place: the texture named in the packet is both what it reads and
	// what it writes. It also returns before it is finished -- "mode async (residual from an
	// earlier frame)" -- so the packet's texture must not be the one the picture is built
	// from, or the next frame overwrites a frame still being read. One set for the frame
	// being recorded, one holding the frame that finished.
	//
	// `full` and `baseline` belong to the same generation as the surface they pair with: the
	// resolve subtracts baseline from what the engine produced to isolate the edit, and puts
	// that over the detail in `full`. Mixing generations there would align one frame's edit
	// against another frame's picture.
	winrt::com_ptr<ID3D12Resource> net[kSlots];
	winrt::com_ptr<ID3D12Resource> full[kSlots];
	winrt::com_ptr<ID3D12Resource> baseline[kSlots];
	winrt::com_ptr<ID3D12Resource> resolved;
	winrt::com_ptr<ID3D12Resource> motion[kSlots];
	winrt::com_ptr<ID3D12Resource> depth[kSlots];
	// One pixel, holding the exposure the engine is told to use instead of adapting one.
	winrt::com_ptr<ID3D12Resource> exposureTexture;
	winrt::com_ptr<ID3D12DescriptorHeap> heap;
	winrt::com_ptr<ID3D12RootSignature> root;
	winrt::com_ptr<ID3D12PipelineState> convertIn;
	winrt::com_ptr<ID3D12PipelineState> convertInSrgb;
	winrt::com_ptr<ID3D12PipelineState> convertOut;
	winrt::com_ptr<ID3D12PipelineState> convertOutSrgb;

	// Whether Magpie's frame is handed to the engine as linear values rather than as the
	// sRGB-encoded ones it arrives in. Read from the ini so it can be A/B'd without a build.
	bool srgbInput = true;




	// Whether the engine's edited frame is handed to the FSR3 upscaler to reconstruct, rather
	// than composited back here as a residual. Read from the ini like srgbInput and editBound,
	// so the two routes can be compared without a rebuild.
	bool reconstruct = false;

	// The anchor of the output shoulder, in thousandths of linear. Lower preserves more
	// highlight structure at the cost of dimming the top of the range; the reason it is a
	// setting rather than a constant is on the conversion shader.
	uint32_t shoulderMilli = 850;

	// Only so the engine-value line is printed when something changes rather than per frame.
	int loggedStyle = -1;
	float loggedIntensity = -1.0f;

	// How large an edit the resolve will apply, in thousandths of the local magnitude.
	// Lower is calmer; the reason it is a setting rather than a constant is on the shader.
	uint32_t editBoundMilli = 500;

	// The reconstruction route.
	//
	// Everything this backend does after the engine has edited its frame is a way of getting
	// the edit back onto the captured picture: a residual taken at the network's resolution,
	// reprojected, and composited onto the full-resolution frame. This project already carries
	// a finished upscaler that does that job with a temporal accumulator behind it --
	// FSR3Upscaler takes a render-resolution colour, copies it into shared surfaces of its
	// own, takes motion from the frame guidance, supplies its own flat depth, runs at zero
	// jitter and writes the upscaled result. Handing it the engine's edited frame replaces the
	// resolve and the two hand-written temporal passes with an accumulator built for this.
	//
	// What that is not, plainly: the captured frame has already been upscaled by the game, so
	// the colour handed over carries no sub-pixel jitter and there is no depth to give. An
	// implementation sitting before the game's upscaler has both and reconstructs better for
	// it. This is the same accumulator working from less.
#ifdef MP_ENABLE_FSR3_ZEROMV
	// The concrete type, not the interface: the HDR protocol is set on the class and the
	// interface has no virtual for it.
	std::unique_ptr<FSR3Upscaler> reconstruction;
#else
	std::unique_ptr<NativeEffectBackend> reconstruction;
#endif
	winrt::com_ptr<ID3D11Texture2D> netShared11;
	winrt::com_ptr<ID3D12Resource> netShared12;
	// Where the encode lands before it crosses. A texture this device owns outright, so the
	// encoding pass writes somewhere no other device has an interest in; the crossing is a
	// plain copy afterwards, which is the shape this backend has always used and the only one
	// it has never taken the device down with.
	//
	// Encoding straight into the shared texture was the first attempt, and it removed the
	// device every time. That surface is not inert: the upscaler holds it and reads it as a
	// shader resource on its own device, while this queue turns it into a UAV to write it.
	// One surface in both roles at once is what the driver refused.
	winrt::com_ptr<ID3D12Resource> encoded;
	// Motion at the network's extent, for the upscaler to reproject with.
	//
	// It asks the frame guidance for motion, and it checks that what it is handed matches its
	// own input extent -- depth, motion and confidence all present and all at that size. The
	// guidance this backend is given is produced at the captured extent, so it cannot be
	// passed through; what goes over instead is a view built here, at the network's size,
	// carrying the motion this backend already resamples to exactly that extent.

	winrt::com_ptr<ID3D11Texture2D> motionShared11;
	winrt::com_ptr<ID3D12Resource> motionShared12;
	// The three the check needs and the dispatch does not read: the upscaler supplies its own
	// flat depth and never samples the ones handed in, so these exist to satisfy the contract
	// and are never written after creation. The zero motion is the honest answer for a frame
	// whose own motion is not ready.
	winrt::com_ptr<ID3D11Texture2D> zeroMotion11;
	winrt::com_ptr<ID3D11Texture2D> guideDepth11;
	winrt::com_ptr<ID3D11Texture2D> guideConfidence11;
	uint32_t reconstructionDraws = 0;
	uint32_t reconstructionFailures = 0;

	winrt::com_ptr<ID3D12PipelineState> temporal;

	// Anti-flicker: the residual carried between frames at the network's resolution, with
	// the guide it was measured against. Ping-ponged, because the pass reads one pair and
	// writes the other. Alpha in both carries validity, not opacity.
	winrt::com_ptr<ID3D12Resource> historyResidual[2];
	winrt::com_ptr<ID3D12Resource> historyGuide[2];
	uint32_t historyIndex = 0;

	// The blend weight comes from the capture interval, so the accumulation is a duration in
	// real time rather than a frame count -- upstream derives it the same way,
	// exp(-dt/0.08) with anything past 0.25 s treated as stale. A repeated capture is not a
	// new sample, and neither is a frame the effect was rebuilt for.
	uint64_t lastFrameId = 0;
	uint64_t lastRevision = 0;
	uint64_t lastTimestamp = 0;
	bool temporalValid = false;

	// The effect's own parameters, kept so the per-pass block can drive the engine from
	// them. Magpie rebuilds the effect when they change, so this is how new values arrive.
	DLSSNRSettings settings{};

	// What the effect asked for. Zero leaves the residual pass subtracting and nothing else,
	// which is exactly what ran before any of this existed. One is static accumulation, two
	// and above reproject the history with optical flow.
	int antiFlickerMode = 0;

	// Optical flow, supplied by Magpie's provider and opened into this device through the
	// same interop the NGX path uses. The engine is a temporal network: with no motion it
	// aligns its history blindly wherever anything moves, which is what the residual flicker
	// looked like before any of this existed.
	std::unique_ptr<FrameGuidanceD3D12Interop> guidanceInterop;
	winrt::com_ptr<ID3D12PipelineState> motionResample;
	bool motionReady = false;
	float motionScaleX = 1.0f, motionScaleY = 1.0f;
	bool wantMotion = false;
	bool timerRaised = false;
	// What this backend asks the renderer's guidance service for, mirrored back at it so the
	// optical-flow provider actually runs for us.
	MotionVectorRequest motionRequest{};

	// Whether the network runs below the capture size. Not the same as the resolve path
	// being in use: anti-flicker needs the resolve even at 100%.
	bool downscaling = false;
	winrt::com_ptr<ID3D12PipelineState> downsample;
	winrt::com_ptr<ID3D12PipelineState> resolve;
	uint32_t descriptorStride = 0;

	// The resolution the network runs at, which is the capture size times modelScale.
	// `scaled` is false at 100%, and then the whole downsample/resolve pair is skipped and
	// the engine works on `full` directly -- the path that shipped before this existed.
	uint32_t netWidth = 0, netHeight = 0;
	float modelScale = 1.0f;
	bool scaled = false;

	HMODULE runtime = nullptr;
	HMODULE hip = nullptr;
	HipSetFn hipSet = nullptr;
	int hipDevice = -1;
	InitFn init = nullptr;
	RecordFn record = nullptr;
	NotifyFn notify = nullptr;
	// Which build of the runtime `runtime` is. Identified from its hash at load time and used
	// for every address taken into its image, so there is no path that reads it before it is
	// set; LoadRuntime refuses the load when it cannot be identified.
	const RuntimeOffsets* rva = nullptr;
	bool ready = false;
	bool failed = false;
	// The engine is brought up once per process, like the runtime module it lives in and
	// unlike every other resource this backend owns. Set at the end of InitEngine and never
	// cleared, because nothing here tears the engine down. See the note there.
	bool engineUp = false;
	// The engine loads kernels and builds its pipeline on the first job or two, so
	// the first frames legitimately take far longer than the steady state. A single
	// slow frame is a frame to skip, not a reason to disable the effect for good.
	uint32_t framesSeen = 0;
	uint32_t timeouts = 0;
	// A one-shot readback used to find out which stage of the chain is producing an
	// empty picture. Reading a texture back is far too slow for every frame, but once,
	// on the first, it turns a guess into a measurement.
	winrt::com_ptr<ID3D12Resource> probe;
	uint64_t probeBytes = 0;
	bool probed = false;
	bool profiled = false;
	// The previous call's samples, for the flicker measure. Never used for anything else.
	std::vector<float> sample;
	std::vector<float> previousSample;
	// TEMPORARY DIAGNOSTIC: how many frames of the series have been logged.
	int probeSamples = 0;

	// Where the frame's time goes, on the performance counter.
	//
	// The engine reports its own job in whole milliseconds and, by the shape of the numbers
	// it prints, off a clock that only moves about every 16 ms: every job it has ever logged
	// here is 15, 16, 31, 32, 46, 47, 62, 63 or 78 ms and nothing in between. That is enough
	// to see a job is slower at 1080p than at 540p, and nothing like enough to say which
	// stage of the frame the time is in, which is the only question worth asking before
	// optimising anything. These are the same stages, timed here.
	uint64_t phaseAccum[7] = {};
	uint64_t phaseAt[7] = {};

	// Where the composite stage's own time goes. That stage is the whole frame here, and it
	// holds two separate blocking waits plus the passes between them, so the stage total
	// cannot say which of them is the cost. Indices: out-wait, temporal, resolve, submit,
	// drain.
	uint64_t compositeAccum[5] = {};

	// Frame-to-frame intervals, for the window the phase log covers.
	//
	// The mean is the number everyone quotes and the one that says least about how a frame
	// rate feels: a run alternating 8 ms and 60 ms has the same mean as a steady 34, and
	// looks nothing like it. The percentiles and the worst frame are what say that, and they
	// are also the only place a hitch shows up at all -- the engine's watchdog, a capture
	// gap, a fence that took a second. The phases measure what this backend spends inside
	// the frame; these measure the frame.
	float frameIntervalsMs[128] = {};
	uint32_t frameIntervalCount = 0;
	uint64_t lastDrawQpc = 0;

	// The frame that has just finished, kept so that a slow one can be reported with its own
	// phases rather than the next frame's: the interval that says a frame was slow is only
	// known when the following frame starts, by which point the slow frame's marks are gone.
	float prevFramePhasesMs[6] = {};
	float prevFrameTotalMs = 0.0f;
	FrameGuidanceFrameId prevFrameId = 0;
	uint32_t prevFrameJob = 0;
	uint32_t slowFramesLogged = 0;
	uint32_t phaseFrames = 0;
	uint64_t phaseWindowStart = 0;

	uint32_t width = 0, height = 0;

	// The frame counter the phase log runs on, so a reader can tell it from the engine's.
	static uint64_t Qpc() noexcept {
		LARGE_INTEGER v{};
		QueryPerformanceCounter(&v);
		return static_cast<uint64_t>(v.QuadPart);
	}

	static double MsPerTick() noexcept {
		static const double scale = [] {
			LARGE_INTEGER f{};
			QueryPerformanceFrequency(&f);
			return f.QuadPart ? 1000.0 / static_cast<double>(f.QuadPart) : 0.0;
		}();
		return scale;
	}

	~Impl() {
		// The session's history storage is about to be released with the members below, and the
		// runtime is still holding a pointer to it. Said here, where the storage goes, rather
		// than left to the next session's first submission -- see ForgetHistory.
		ForgetHistory();
		// The runtime stays loaded for the life of the process: it starts worker threads
		// holding references into its own image, and unloading it under them is not
		// something this backend can make safe.
		if (timerRaised) {
			timeEndPeriod(1);
		}
	}

	D3D12_GPU_DESCRIPTOR_HANDLE Gpu(uint32_t index) const noexcept {
		D3D12_GPU_DESCRIPTOR_HANDLE h = heap->GetGPUDescriptorHandleForHeapStart();
		h.ptr += UINT64(index) * descriptorStride;
		return h;
	}

	D3D12_CPU_DESCRIPTOR_HANDLE Cpu(uint32_t index) const noexcept {
		D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
		h.ptr += SIZE_T(index) * descriptorStride;
		return h;
	}

	bool LoadRuntime() noexcept;
	bool CreateHip() noexcept;
	bool CreatePipeline() noexcept;
	bool CreateSized(uint32_t w, uint32_t h, DXGI_FORMAT inFmt, DXGI_FORMAT outFmt) noexcept;
	bool InitEngine(const std::filesystem::path& weightsPath) noexcept;
	bool CreateExposure() noexcept;
	void Bind(uint32_t tableSlot, ID3D12Resource* srv, DXGI_FORMAT srvFormat,
		ID3D12Resource* uav, DXGI_FORMAT uavFormat) noexcept;
	void BindResolve(uint32_t tableSlot, ID3D12Resource* srv, ID3D12Resource* uav,
		ID3D12Resource* baselineTexture, ID3D12Resource* edited) noexcept;
	void BindTemporal(uint32_t tableSlot, ID3D12Resource* edited, ID3D12Resource* nextResidual,
		ID3D12Resource* nextGuide, ID3D12Resource* baselineTexture,
		ID3D12Resource* historyResidualTexture,
		ID3D12Resource* historyGuideTexture, ID3D12Resource* motionTexture) noexcept;
	void CreateSrv(ID3D12Resource* res, D3D12_CPU_DESCRIPTOR_HANDLE where) noexcept;
	bool RunTemporal(const NativeEffectDrawContext& context) noexcept;
	void Dispatch(ID3D12PipelineState* pso, uint32_t tableSlot) noexcept;
	// `residualTableSlot` is where the second descriptor table starts, or 0 for a pass that
	// has no second table. It is a slot rather than a flag because the passes lay their
	// descriptors out differently: the temporal pass writes two UAVs before its extra
	// sources, the resolve writes one.
	void DispatchSized(ID3D12PipelineState* pso, uint32_t tableSlot, uint32_t dw, uint32_t dh,
		uint32_t residualTableSlot, const UINT* constants = nullptr,
		uint32_t constantCount = 4) noexcept;
	// Waits until the engine reports it has finished job number `wanted`. The counter is the
	// engine's own published progress, so it is global rather than per slot -- the job number
	// is what says which frame has landed.
	bool WaitForEngine(uint32_t wanted, uint64_t deadlineMs) noexcept;
	// Frees every slot whose work has both finished on the engine and retired on this queue.
	// The reference runs this after every submission and on every status query; it is what
	// turns a slot from "in flight" back into "available", and it is the only place a slot
	// becomes reusable.
	void RetireSubmission(const char* source) noexcept;
	// One-off check of the device boundary in both directions. Diagnostic only.
	void CrossTest() noexcept;
	bool SubmitEngineJob() noexcept;
	// Brings up the FSR3 upscaler over a shared copy of the engine's frame, when the route is
	// selected. A no-op otherwise, and a no-op when the SDK is not in this build.
	bool CreateReconstruction(DeviceResources& resources, ID3D11Texture2D* output) noexcept;
	// Hands the engine's edited frame to it and lets it write the output.
	bool RunReconstruction(const NativeEffectDrawContext& context) noexcept;
	void DestroySized() noexcept;
	// Tells the runtime that the history it was carrying is gone. Called where the storage
	// actually goes away -- see the definition.
	void ForgetHistory() noexcept;
	// What actually reached the display, read back once. The route's whole claim is that the
	// picture is display-referred -- that the shoulder and the encode happen before the
	// reconstruction rather than after it, which is what the darker version of this route got
	// wrong. This is how that claim is checked without a person looking at a screen.
	float MeanOfOutput(ID3D11Texture2D* tex, float* detail = nullptr) noexcept;

	float Measure(ID3D12Resource* res, DXGI_FORMAT format,
		D3D12_RESOURCE_STATES before, uint64_t* nonFinite = nullptr,
		float* meanAbsDelta = nullptr) noexcept;
};

bool DlssnrAmdBackend::Impl::LoadRuntime() noexcept {
	if (runtime) {
		return true;
	}

	EngineHost& host = Engine();
	if (host.runtime) {
		// An earlier session in this process already loaded and identified the module. The
		// runtime is never unloaded, so there is nothing to load: this only has to pick up the
		// handle and the offsets.
		//
		// This is reached from Initialize on every session, which matters because that is where
		// a session would stop with nothing more in the log. A dump taken from one of those had
		// the loader's own path in KERNELBASE on the stack, entering the runtime's image, on the
		// thread that was already inside it -- so re-entering the loader here, for a module this
		// process holds and will hold until it exits, is worth not doing.
		runtime = host.runtime;
		rva = host.rva;
	} else {
		const auto dir = ExeDirectory();
		const auto runtimePath = dir / kRuntimeName;
		const auto weightsPath = dir / kWeightsName;

		std::error_code ec;
		if (!std::filesystem::exists(runtimePath, ec) ||
			!std::filesystem::exists(weightsPath, ec)) {
			return false;
		}
		rva = IdentifyRuntime(runtimePath);
		if (!rva) {
			Logger::Get().Error(
				"DLSSNR AMD: dlssnr_amd_pass1.dll is neither of the builds this backend carries "
				"offsets for; refusing rather than guessing at offsets into a different layout");
			return false;
		}

		// The engine reads this from DllMain, so it must exist first. It is left empty: every
		// key that could go in it overrides a default that is already correct, and the
		// working installation ships it at zero bytes.
		const auto iniPath = dir / kIniName;
		if (!std::filesystem::exists(iniPath, ec)) {
			wil::unique_hfile ini(CreateFileW(iniPath.c_str(), GENERIC_WRITE, 0, nullptr,
				CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
			if (!ini) {
				return false;
			}
		}

		runtime = LoadLibraryExW(runtimePath.c_str(), nullptr,
			LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
		if (!runtime) {
			Logger::Get().Error(fmt::format("DLSSNR AMD: LoadLibrary failed (error {})",
				GetLastError()));
			return false;
		}
		host.runtime = runtime;
		host.rva = rva;
	}

	init = reinterpret_cast<InitFn>(reinterpret_cast<uintptr_t>(runtime) + rva->init);
	record = reinterpret_cast<RecordFn>(reinterpret_cast<uintptr_t>(runtime) + rva->record);
	notify = reinterpret_cast<NotifyFn>(reinterpret_cast<uintptr_t>(runtime) + rva->notify);
	return true;
}

bool DlssnrAmdBackend::Impl::CreateHip() noexcept {
	if (hipSet) {
		return true;
	}
	hip = LoadLibraryExW(L"amdhip64_7.dll", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
	if (!hip) {
		Logger::Get().Error("DLSSNR AMD: amdhip64_7.dll is not available");
		return false;
	}

	auto getCount = reinterpret_cast<int (*)(int*)>(GetProcAddress(hip, "hipGetDeviceCount"));
	auto getProps = reinterpret_cast<int (*)(void*, int)>(
		GetProcAddress(hip, "hipGetDevicePropertiesR0600"));
	hipSet = reinterpret_cast<HipSetFn>(GetProcAddress(hip, "hipSetDevice"));
	if (!getCount || !getProps || !hipSet) {
		return false;
	}

	int count = 0;
	if (getCount(&count) != 0 || count <= 0) {
		return false;
	}

	// The engine's own inline diagnostic reports "HIP runtime 0" where a working system reports
	// 70260201, and it reads that as a driver too old to be supported -- which is the reason it
	// gives for the inline wait timing out. Asking the same question here says whether the
	// reading is right, or whether it is an artefact of when the engine asks.
	{
		auto runtimeVersion = reinterpret_cast<int (*)(int*)>(
			GetProcAddress(hip, "hipRuntimeGetVersion"));
		auto driverVersion = reinterpret_cast<int (*)(int*)>(
			GetProcAddress(hip, "hipDriverGetVersion"));
		int runtimeAnswer = -1;
		int driverAnswer = -1;
		const int runtimeCall = runtimeVersion ? runtimeVersion(&runtimeAnswer) : -99;
		const int driverCall = driverVersion ? driverVersion(&driverAnswer) : -99;
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD: HIP runtime version {} (call {}), driver version {} (call {}), "
			"devices {}",
			runtimeAnswer, runtimeCall, driverAnswer, driverCall, count));
	}

	// The D3D12 device has to be on the same physical card as the HIP kernels, or the
	// textures handed to the engine live in memory it cannot reach. hipDeviceProp_tR0600
	// carries the adapter LUID for exactly this comparison -- and it is around a
	// kilobyte, so all of it has to be provided. A smaller buffer is an ordinary stack
	// overrun, not an error return.
	const LUID target = device12->GetAdapterLuid();
	struct { alignas(8) unsigned char raw[4096]; } props{};
	for (int i = 0; i < count; ++i) {
		if (getProps(&props, i) != 0) {
			continue;
		}
		if (std::memcmp(props.raw + 272, &target, sizeof(target)) == 0) {
			hipDevice = i;
			break;
		}
	}
	if (hipDevice < 0) {
		// Which adapter the renderer is on, and which ones HIP is offering. This machine has
		// two, and when the renderer lands on the one without a HIP device the backend cannot
		// start at all -- a failure that reads as the effect being broken and is not.
		std::string seen;
		for (int i = 0; i < count; ++i) {
			if (getProps(&props, i) != 0) {
				continue;
			}
			LUID luid{};
			std::memcpy(&luid, props.raw + 272, sizeof(luid));
			seen += fmt::format("{}{:08x}:{:08x}", seen.empty() ? "" : ", ",
				static_cast<uint32_t>(luid.HighPart), static_cast<uint32_t>(luid.LowPart));
		}
		Logger::Get().Error(fmt::format(
			"DLSSNR AMD: no HIP device matches the render adapter -- renderer is on LUID "
			"{:08x}:{:08x}, HIP offers [{}] of {} device(s)",
			static_cast<uint32_t>(target.HighPart), static_cast<uint32_t>(target.LowPart),
			seen, count));
		return false;
	}
	if (hipSet(hipDevice) != 0) {
		return false;
	}
	return true;
}

void DlssnrAmdBackend::Impl::RetireSubmission(const char* source) noexcept {
	(void)source;
	const uint64_t completed = fence ? fence->GetCompletedValue() : 0;
	for (uint32_t k = 0; k < kSlots; ++k) {
		SlotState& st = slotState[k];
		if (!st.recorded) {
			continue;
		}
		// The engine's own completion for this slot. With the asynchronous handshake it does
		// not advance, and that is not a reason to keep the slot forever -- the queue's fence
		// is the authority here, exactly as it is for the picture.
		const uint64_t target = slotFence[k];
		if (st.CanRetire(true, completed, target)) {
			st = {};
		}
	}
}

void DlssnrAmdBackend::Impl::CrossTest() noexcept {
	// The crossing, both ways, with a value nothing can fake. D3D12 writes a constant into the
	// shared output surface, then D3D11 reads it back with the measure that is known to work.
	// One direction working says which side is broken; neither working says the shared surface
	// is not doing what it is supposed to do. Run once, outside the frame loop, so it cannot
	// disturb the per-frame fence accounting.
	if (!crossTestList) {
		const HRESULT h1 = device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
			IID_PPV_ARGS(crossTestAllocator.put()));
		const HRESULT h2 = SUCCEEDED(h1) ? device12->CreateCommandList(0,
			D3D12_COMMAND_LIST_TYPE_DIRECT, crossTestAllocator.get(), nullptr,
			IID_PPV_ARGS(crossTestList.put())) : h1;
		if (FAILED(h1) || FAILED(h2)) {
			Logger::Get().Error(fmt::format("CROSS TEST: list creation failed 0x{:08x}/0x{:08x}",
				static_cast<uint32_t>(h1), static_cast<uint32_t>(h2)));
			return;
		}
		crossTestList->Close();
	}
	if (crossTestFenceValue == 0) {
		const HRESULT hf = device12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
			IID_PPV_ARGS(crossTestFence.put()));
		if (FAILED(hf)) {
			Logger::Get().Error(fmt::format("CROSS TEST: fence failed 0x{:08x}",
				static_cast<uint32_t>(hf)));
			return;
		}
	}
	const HRESULT hr1 = crossTestAllocator->Reset();
	const HRESULT hr2 = SUCCEEDED(hr1)
		? crossTestList->Reset(crossTestAllocator.get(), nullptr) : hr1;
	if (FAILED(hr1) || FAILED(hr2)) {
		Logger::Get().Error(fmt::format("CROSS TEST: reset failed 0x{:08x}/0x{:08x}",
			static_cast<uint32_t>(hr1), static_cast<uint32_t>(hr2)));
		return;
	}
	// Closed straight away because nothing below records into it: this function does all its
	// work on lists of its own. An allocator cannot be reset while a list built from it is
	// still open, so leaving this one open made the second call -- the one that runs after the
	// runtime has installed its hooks, which is the call that matters -- fail before it began.
	crossTestList->Close();
	const uint32_t descSlot = 30;
	D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
	u.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	// A brand new pair at a small size, so nothing about the large surfaces or their reuse can
	// be blamed. This is the smallest possible statement of "can these two devices share".
	winrt::com_ptr<ID3D11Texture2D> small11;
	winrt::com_ptr<ID3D12Resource> small12;
	D3D11_TEXTURE2D_DESC sd{};
	sd.Width = 64; sd.Height = 64; sd.MipLevels = 1; sd.ArraySize = 1;
	sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.SampleDesc.Count = 1;
	sd.Usage = D3D11_USAGE_DEFAULT;
	sd.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	sd.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
	// A second, entirely independent D3D11 device on the same adapter. If a texture made and
	// filled on THAT device is visible to D3D12 while Magpie's own is not, the problem is in
	// how the renderer's device was created; if neither is visible, it is the platform.
	winrt::com_ptr<ID3D11Device> probe11;
	winrt::com_ptr<ID3D11DeviceContext> probeDC;
	winrt::com_ptr<ID3D11Texture2D> probeTex;
	bool probeOk = false;
	{
		D3D_FEATURE_LEVEL fl{};
		const D3D_FEATURE_LEVEL want[]{ D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
		// Found by enumeration, the way the standalone test that works does it, rather than
		// through the renderer's device: the point is a device this backend did not influence.
		winrt::com_ptr<IDXGIFactory4> fac;
		winrt::com_ptr<IDXGIAdapter1> pick;
		if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(fac.put())))) {
			for (UINT i = 0; fac->EnumAdapters1(i, pick.put()) == S_OK; ++i) {
				DXGI_ADAPTER_DESC1 d{};
				pick->GetDesc1(&d);
				if (d.VendorId == 0x1002) break;
				pick = nullptr;
			}
		}
		HRESULT hprobe = E_FAIL;
		if (pick) {
			hprobe = D3D11CreateDevice(pick.get(), D3D_DRIVER_TYPE_UNKNOWN,
				nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, want, 2, D3D11_SDK_VERSION,
				probe11.put(), &fl, probeDC.put());
		}
		Logger::Get().Info(fmt::format("CROSS TEST: probe device hr=0x{:08x} adapter={}",
			static_cast<uint32_t>(hprobe), pick ? "found" : "NONE"));
		if (pick && SUCCEEDED(hprobe)) {
			D3D11_TEXTURE2D_DESC pd = sd;
			if (SUCCEEDED(probe11->CreateTexture2D(&pd, nullptr, probeTex.put()))) {
				// Filled through a staging copy, the way the standalone test does it: a UAV-capable
				// texture cannot take initial data directly.
				D3D11_TEXTURE2D_DESC st = sd;
				st.Usage = D3D11_USAGE_STAGING;
				st.BindFlags = 0;
				st.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				st.MiscFlags = 0;
				winrt::com_ptr<ID3D11Texture2D> stage;
				if (SUCCEEDED(probe11->CreateTexture2D(&st, nullptr, stage.put()))) {
					D3D11_MAPPED_SUBRESOURCE m{};
					if (SUCCEEDED(probeDC->Map(stage.get(), 0, D3D11_MAP_WRITE, 0, &m))) {
						for (UINT y = 0; y < st.Height; ++y) {
							uint8_t* row = static_cast<uint8_t*>(m.pData) + size_t(y) * m.RowPitch;
							for (UINT x = 0; x < st.Width; ++x) {
								row[x * 4 + 0] = 128;
								row[x * 4 + 1] = 128;
								row[x * 4 + 2] = 128;
								row[x * 4 + 3] = 255;
							}
						}
						probeDC->Unmap(stage.get(), 0);
						probeDC->CopyResource(probeTex.get(), stage.get());
						probeDC->Flush();
						probeOk = true;
					}
				}
			}
		}
	}
	Logger::Get().Info(fmt::format("CROSS TEST: independent D3D11 device {}", probeOk ? "ok" : "FAILED"));
	// If it exists, run the same crossing on IT. A texture made and filled on a device this
	// backend did not create, read back on D3D12, separates "the renderer's device" from
	// "the platform" with no other variable in play.
	if (probeOk) {
		winrt::com_ptr<IDXGIResource1> pxr;
		HANDLE ph = nullptr;
		winrt::com_ptr<ID3D12Resource> probe12;
		bool shared = false;
		if (SUCCEEDED(probeTex->QueryInterface(IID_PPV_ARGS(pxr.put()))) &&
			SUCCEEDED(pxr->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &ph))) {
			wil::unique_handle phh(ph);
			if (SUCCEEDED(device12->OpenSharedHandle(phh.get(), IID_PPV_ARGS(probe12.put())))) {
				shared = true;
			}
		}
		Logger::Get().Info(fmt::format(
			"CROSS TEST: independent device's texture opened by D3D12: {}",
			shared ? "yes" : "NO"));
		// The full round trip on the independent pair: D3D12 writes a known value into the
		// independent device's shared texture, that device reads it back. Magpie's own devices
		// and contexts are not involved at any point, so a failure here can only be about the
		// D3D12 device this backend created.
		{
			winrt::com_ptr<ID3D12Resource> indep12;
			winrt::com_ptr<IDXGIResource1> ixr;
			// A queue of its own, so nothing submitted here goes through the engine's
			// interception of the backend's own queue.
			winrt::com_ptr<ID3D12CommandQueue> altQueue;
			winrt::com_ptr<ID3D12CommandAllocator> altAlloc;
			winrt::com_ptr<ID3D12GraphicsCommandList> altList;
			winrt::com_ptr<ID3D12Fence> altFence;
			{
				D3D12_COMMAND_QUEUE_DESC aqd{};
				aqd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
				device12->CreateCommandQueue(&aqd, IID_PPV_ARGS(altQueue.put()));
				device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
					IID_PPV_ARGS(altAlloc.put()));
				device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
					altAlloc.get(), nullptr, IID_PPV_ARGS(altList.put()));
				device12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(altFence.put()));
			}
			HANDLE ih = nullptr;
			bool roundTripRan = false;
			if (SUCCEEDED(probeTex->QueryInterface(IID_PPV_ARGS(ixr.put()))) &&
				SUCCEEDED(ixr->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &ih))) {
				wil::unique_handle ihh(ih);
				if (SUCCEEDED(device12->OpenSharedHandle(ihh.get(),
						IID_PPV_ARGS(indep12.put())))) {
					D3D12_UNORDERED_ACCESS_VIEW_DESC iu{};
					iu.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
					iu.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
					device12->CreateUnorderedAccessView(indep12.get(), nullptr, &iu, Cpu(descSlot));
					Barrier(altList.get(), indep12.get(), D3D12_RESOURCE_STATE_COMMON,
						D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
					const float kv[4]{ 0.25f, 0.25f, 0.25f, 1.0f };
					altList->ClearUnorderedAccessViewFloat(Gpu(descSlot), Cpu(descSlot),
						indep12.get(), kv, 0, nullptr);
					Barrier(altList.get(), indep12.get(),
						D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
					altList->Close();
					ID3D12CommandList* icl[]{ altList.get() };
					altQueue->ExecuteCommandLists(1, icl);
					const uint64_t isig = ++crossTestFenceValue;
					altQueue->Signal(altFence.get(), isig);
					altFence->SetEventOnCompletion(isig, crossTestEvent);
					WaitForSingleObject(crossTestEvent, 3000);
					roundTripRan = true;
					altAlloc->Reset();
					altList->Reset(altAlloc.get(), nullptr);
				}
			}
			probeDC->Flush();
			Logger::Get().Info(fmt::format(
				"CROSS TEST: independent pair round trip ran={}", roundTripRan));
			D3D11_TEXTURE2D_DESC rs = sd;
			rs.Usage = D3D11_USAGE_STAGING;
			rs.BindFlags = 0;
			rs.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			rs.MiscFlags = 0;
			winrt::com_ptr<ID3D11Texture2D> rst;
			if (SUCCEEDED(probe11->CreateTexture2D(&rs, nullptr, rst.put()))) {
				probeDC->CopyResource(rst.get(), probeTex.get());
				D3D11_MAPPED_SUBRESOURCE rm{};
				if (SUCCEEDED(probeDC->Map(rst.get(), 0, D3D11_MAP_READ, 0, &rm))) {
					const uint8_t* rp = static_cast<const uint8_t*>(rm.pData);
					Logger::Get().Info(fmt::format(
						"CROSS TEST: independent device read = {} (128 = untouched, 64 = D3D12's "
						"write arrived)", rp[0]));
					probeDC->Unmap(rst.get(), 0);
				}
			}
		}
	}

	if (FAILED(device11->CreateTexture2D(&sd, nullptr, small11.put()))) {
		Logger::Get().Error("CROSS TEST: small texture creation failed");
		return;
	}
	winrt::com_ptr<IDXGIResource1> xr;
	HANDLE h = nullptr;
	if (FAILED(small11->QueryInterface(IID_PPV_ARGS(xr.put()))) ||
		FAILED(xr->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &h))) {
		Logger::Get().Error("CROSS TEST: small texture could not be shared");
		return;
	}
	{
		wil::unique_handle hh(h);
		if (FAILED(device12->OpenSharedHandle(hh.get(), IID_PPV_ARGS(small12.put())))) {
			Logger::Get().Error("CROSS TEST: D3D12 could not open the small texture");
			return;
		}
	}
	// A queue of our own, made here and used only here. The engine hooks the queue this
	// backend runs on, so everything submitted through that one goes through the engine's
	// interception -- if the write below only lands on this fresh queue, that interception
	// is what the crossing has been fighting all along.
	winrt::com_ptr<ID3D12CommandQueue> altQueue;
	winrt::com_ptr<ID3D12CommandAllocator> altAlloc;
	winrt::com_ptr<ID3D12GraphicsCommandList> altList;
	winrt::com_ptr<ID3D12Fence> altFence;
	{
		D3D12_COMMAND_QUEUE_DESC aqd{};
		aqd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
		if (FAILED(device12->CreateCommandQueue(&aqd, IID_PPV_ARGS(altQueue.put()))) ||
			FAILED(device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
				IID_PPV_ARGS(altAlloc.put()))) ||
			FAILED(device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
				altAlloc.get(), nullptr, IID_PPV_ARGS(altList.put()))) ||
			FAILED(device12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
				IID_PPV_ARGS(altFence.put())))) {
			Logger::Get().Error("CROSS TEST: fresh queue unavailable");
			return;
		}
	}
	device12->CreateUnorderedAccessView(small12.get(), nullptr, &u, Cpu(descSlot));
	Barrier(altList.get(), small12.get(), D3D12_RESOURCE_STATE_COMMON,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	const float known[4]{ 0.5f, 0.5f, 0.5f, 1.0f };
	altList->ClearUnorderedAccessViewFloat(Gpu(descSlot), Cpu(descSlot),
		small12.get(), known, 0, nullptr);
	Barrier(altList.get(), small12.get(),
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
	altList->Close();
	ID3D12CommandList* cl[]{ altList.get() };
	altQueue->ExecuteCommandLists(1, cl);
	const uint64_t sig = ++crossTestFenceValue;
	altQueue->Signal(altFence.get(), sig);
	if (!crossTestEvent) {
		crossTestEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	}
	if (altFence->GetCompletedValue() < sig) {
		altFence->SetEventOnCompletion(sig, crossTestEvent);
		WaitForSingleObject(crossTestEvent, 2000);
	}
	// The other direction, for contrast: D3D11 fills it, D3D12 reads it back.
	// 'small' is a macro in windef.h (char); the name is avoided deliberately.
	const float smallMean = MeanOfOutput(small11.get());
	Logger::Get().Info(fmt::format(
		"DLSSNR AMD CROSS TEST (fresh queue): D3D12 wrote 0.5, D3D11 reads {:.4f}",
		smallMean));

	// ---- the instrument, and the direction the backend actually runs on ----
	//
	// Everything the profile reports on the D3D12 side goes through Measure, and Measure
	// submits on the queue the engine intercepts. So a reading of zero has two possible
	// authors -- the chain produced nothing, or the probe's own copy was the thing that got
	// lost -- and until they are told apart every D3D12 figure in this backend is ambiguous.
	//
	// The first reading below settles the instrument: small12 was just cleared to 0.5 by D3D12
	// itself, so Measure on it can only be wrong if Measure is.
	const float measureOfKnown = Measure(small12.get(), DXGI_FORMAT_R8G8B8A8_UNORM,
		D3D12_RESOURCE_STATE_COMMON);
	Logger::Get().Info(fmt::format(
		"DLSSNR AMD CROSS TEST: Measure on a value D3D12 wrote reads {:.4f} (expect 0.5020) "
		"-- 0 means the probe, not the chain", measureOfKnown));

	// The second is the direction this backend depends on and that nothing above covered:
	// D3D11 fills the surface, D3D12 reads it. Filled through a staging copy, the fill the
	// standalone test proves, so the crossing is the only variable. The D3D11 queue is drained
	// with an event query before the read, which removes ordering as an excuse and leaves only
	// the question of whether the two devices are looking at the same memory.
	{
		D3D11_TEXTURE2D_DESC st = sd;
		st.Usage = D3D11_USAGE_STAGING;
		st.BindFlags = 0;
		st.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		st.MiscFlags = 0;
		winrt::com_ptr<ID3D11Texture2D> stage;
		bool filled = false;
		if (SUCCEEDED(device11->CreateTexture2D(&st, nullptr, stage.put()))) {
			D3D11_MAPPED_SUBRESOURCE m{};
			if (SUCCEEDED(context11->Map(stage.get(), 0, D3D11_MAP_WRITE, 0, &m))) {
				for (UINT y = 0; y < st.Height; ++y) {
					uint8_t* row = static_cast<uint8_t*>(m.pData) + size_t(y) * m.RowPitch;
					for (UINT x = 0; x < st.Width; ++x) {
						row[x * 4 + 0] = 200;
						row[x * 4 + 1] = 200;
						row[x * 4 + 2] = 200;
						row[x * 4 + 3] = 255;
					}
				}
				context11->Unmap(stage.get(), 0);
				context11->CopyResource(small11.get(), stage.get());

				D3D11_QUERY_DESC qd{};
				qd.Query = D3D11_QUERY_EVENT;
				winrt::com_ptr<ID3D11Query> drained;
				if (SUCCEEDED(device11->CreateQuery(&qd, drained.put()))) {
					context11->End(drained.get());
					context11->Flush();
					const ULONGLONG deadline = GetTickCount64() + 2000;
					while (context11->GetData(drained.get(), nullptr, 0, 0) == S_FALSE &&
						GetTickCount64() < deadline) {
						Sleep(0);
					}
					filled = true;
				}
			}
		}
		Logger::Get().Info(fmt::format(
			"CROSS TEST: D3D11 filled the shared surface and drained: {}",
			filled ? "yes" : "NO"));
		if (filled) {
			const float readD11 = MeanOfOutput(small11.get());
			const float readD12 = Measure(small12.get(), DXGI_FORMAT_R8G8B8A8_UNORM,
				D3D12_RESOURCE_STATE_COMMON);
			Logger::Get().Info(fmt::format(
				"DLSSNR AMD CROSS TEST: D3D11 wrote 200 -- D3D11 reads {:.4f}, D3D12 reads {:.4f} "
				"(expect 0.7843 both; D3D11 right and D3D12 0 is the crossing)",
				readD11, readD12));
		}
	}

	// The production surfaces themselves, with the same value.
	//
	// Every pair above is one this function made at 64x64 in R8G8B8A8, and those cross
	// correctly. The pair the backend actually runs on does not -- D3D11 reads the frame out of
	// sharedIn11 while D3D12 reads zero out of the same memory -- and only two things differ
	// between them: the extent and the format. This says which, or that it is neither.
	//
	// Only runs once the surfaces exist, which is why the early call skips it.
	//
	// Read only, never written. An earlier version filled sharedIn11 to have a value nothing
	// could fake, and it proved the point -- the production pair crossed correctly once the
	// measurement state was right -- but it also replaced the frame the profile was about to
	// report on, so the figures around it stopped describing the pipeline. The pair is read as
	// it stands instead: two readings of the same memory, and whether they agree.
	if (sharedIn11 && sharedIn12) {
		const float readD11 = MeanOfOutput(sharedIn11.get());
		const float readD12 = Measure(sharedIn12.get(), inputFormat,
			D3D12_RESOURCE_STATE_COMMON);
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD CROSS TEST (production pair, as it stands): D3D11 reads {:.4f}, "
			"D3D12 reads {:.4f} -- disagreeing means the crossing, not the chain",
			readD11, readD12));
	}

	// Whether D3D12 writes anything at all once the engine is live.
	//
	// Every figure above that reads zero is read back through ID3D12Resource::Map on a readback
	// heap, and every one of them would read zero just as well if the writes were landing
	// somewhere this process can no longer see. This one writes into a surface the D3D11 device
	// shares and reads it there, with the crossing's own fence between the two ends, so neither
	// Map nor the input path can account for a zero.
	//
	// The answer decides whether moving the whole backend onto D3D12 is worth attempting: if
	// D3D12 produces nothing here, it would produce nothing there either.
	// Guarded because the crossing fences do not exist yet at the call made from CreatePipeline;
	// signalling a null fence there ends the initialisation instead of the frame.
	{
		winrt::com_ptr<ID3D11Texture2D> shared11;
		winrt::com_ptr<ID3D12Resource> shared12;
		if (outFence11 && outFence12 && context11x &&
			CreateSharedTexture(device11, device12.get(), 64, 64,
				DXGI_FORMAT_R8G8B8A8_UNORM, true, shared11, shared12)) {
			winrt::com_ptr<ID3D12CommandAllocator> probeAlloc;
			winrt::com_ptr<ID3D12GraphicsCommandList> probeCl;
			if (SUCCEEDED(device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
					IID_PPV_ARGS(probeAlloc.put()))) &&
				SUCCEEDED(device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
					probeAlloc.get(), nullptr, IID_PPV_ARGS(probeCl.put())))) {
				winrt::com_ptr<ID3D12DescriptorHeap> crossHeap;
				D3D12_DESCRIPTOR_HEAP_DESC crossDesc{};
				crossDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
				crossDesc.NumDescriptors = 1;
				crossDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
				if (FAILED(device12->CreateDescriptorHeap(&crossDesc,
						IID_PPV_ARGS(crossHeap.put())))) {
					return;
				}
				const auto crossCpu = crossHeap->GetCPUDescriptorHandleForHeapStart();
				const auto crossGpu = crossHeap->GetGPUDescriptorHandleForHeapStart();
				D3D12_UNORDERED_ACCESS_VIEW_DESC sharedUav{};
				sharedUav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
				sharedUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
				device12->CreateUnorderedAccessView(shared12.get(), nullptr, &sharedUav,
					crossCpu);
				Barrier(probeCl.get(), shared12.get(), D3D12_RESOURCE_STATE_COMMON,
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
				const float clearValue[4]{ 0.5f, 0.5f, 0.5f, 1.0f };
				probeCl->ClearUnorderedAccessViewFloat(crossGpu, crossCpu,
					shared12.get(), clearValue, 0, nullptr);
				Barrier(probeCl.get(), shared12.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
					D3D12_RESOURCE_STATE_COMMON);
				probeCl->Close();
				ID3D12CommandList* ls[]{ probeCl.get() };
				SubmitTo(queue.get(), 1, ls);
				const uint64_t crossing = ++outFenceValue;
				queue->Signal(outFence12.get(), crossing);
				const HRESULT awaited = context11x->Wait(outFence11.get(), crossing);
				const float seen = MeanOfOutput(shared11.get());
				Logger::Get().Info(fmt::format(
					"DLSSNR AMD CROSS TEST (D3D12 cleared 0.5 into a shared surface, D3D11 reads "
					"it): {:.4f} (expect 0.5020; 0 means D3D12 produces nothing at all with the "
					"engine live, and moving the backend onto D3D12 could not help) wait=0x{:08x}",
					seen, static_cast<uint32_t>(awaited)));
			}
		}
	}

	// Which queue stops working.
	//
	// With the readback no longer reuse-buffered, everything measured on D3D12 reads zero once
	// the runtime has initialised -- a value D3D12 writes and D3D12 reads back comes back zero,
	// on a queue made after the fact and on the shared surfaces alike. The engine keeps
	// reporting jobs throughout, because its work runs on HIP and never touches these queues.
	//
	// So the question is which queue is being swallowed. The two submissions below are the same
	// act, differing only in the queue that carries them: the engine's, which is the one the
	// runtime was handed and hooks, and one this backend created for itself. Whichever reads its
	// value back is a queue that still executes, and that decides the fix.
	const auto probeSubmission = [&](ID3D12CommandQueue* q, const char* what) {
		if (!q) {
			Logger::Get().Info(fmt::format(
				"DLSSNR AMD CROSS TEST ({}): no such queue", what));
			return;
		}
		D3D12_HEAP_PROPERTIES hp{};
		hp.Type = D3D12_HEAP_TYPE_DEFAULT;
		D3D12_RESOURCE_DESC rd{};
		rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		rd.Width = 64;
		rd.Height = 64;
		rd.DepthOrArraySize = 1;
		rd.MipLevels = 1;
		rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		rd.SampleDesc.Count = 1;
		rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		winrt::com_ptr<ID3D12Resource> tex;
		if (FAILED(device12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
				D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(tex.put())))) {
			return;
		}
		// A descriptor heap of this test's own. It used to take a slot from the one this backend
		// writes its passes into, and a frame overwrites that slot within forty frames -- so a
		// clear aimed here would land on whatever resource the frame had just described, and the
		// readback would be zero for a reason that has nothing to do with D3D12 executing.
		winrt::com_ptr<ID3D12DescriptorHeap> ownHeap;
		D3D12_DESCRIPTOR_HEAP_DESC hd{};
		hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		hd.NumDescriptors = 1;
		hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		if (FAILED(device12->CreateDescriptorHeap(&hd, IID_PPV_ARGS(ownHeap.put())))) {
			return;
		}
		const auto ownCpu = ownHeap->GetCPUDescriptorHandleForHeapStart();
		const auto ownGpu = ownHeap->GetGPUDescriptorHandleForHeapStart();
		D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
		u.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		device12->CreateUnorderedAccessView(tex.get(), nullptr, &u, ownCpu);
		D3D12_HEAP_PROPERTIES rh{};
		rh.Type = D3D12_HEAP_TYPE_READBACK;
		D3D12_RESOURCE_DESC rdesc{};
		rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		rdesc.Width = 256ull * 64;
		rdesc.Height = 1;
		rdesc.DepthOrArraySize = 1;
		rdesc.MipLevels = 1;
		rdesc.Format = DXGI_FORMAT_UNKNOWN;
		rdesc.SampleDesc.Count = 1;
		rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		winrt::com_ptr<ID3D12Resource> rb;
		winrt::com_ptr<ID3D12CommandAllocator> al;
		winrt::com_ptr<ID3D12GraphicsCommandList> cl;
		winrt::com_ptr<ID3D12Fence> fe;
		if (FAILED(device12->CreateCommittedResource(&rh, D3D12_HEAP_FLAG_NONE, &rdesc,
				D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(rb.put()))) ||
			FAILED(device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
				IID_PPV_ARGS(al.put()))) ||
			FAILED(device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al.get(),
				nullptr, IID_PPV_ARGS(cl.put()))) ||
			FAILED(device12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fe.put())))) {
			return;
		}
		Barrier(cl.get(), tex.get(), D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		const float known[4]{ 0.5f, 0.5f, 0.5f, 1.0f };
		cl->ClearUnorderedAccessViewFloat(ownGpu, ownCpu, tex.get(), known, 0, nullptr);
		Barrier(cl.get(), tex.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			D3D12_RESOURCE_STATE_COPY_SOURCE);
		D3D12_TEXTURE_COPY_LOCATION src{};
		src.pResource = tex.get();
		src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		D3D12_TEXTURE_COPY_LOCATION dst{};
		dst.pResource = rb.get();
		dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
		dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		dst.PlacedFootprint.Footprint.Width = 64;
		dst.PlacedFootprint.Footprint.Height = 64;
		dst.PlacedFootprint.Footprint.Depth = 1;
		dst.PlacedFootprint.Footprint.RowPitch = 256;
		cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
		cl->Close();
		ID3D12CommandList* submitted[]{ cl.get() };
		// Through the saved entry, so this measures the queue's execution rather than the
		// interception of it: a submission made the ordinary way here would be testing the
		// hook, which is the thing under suspicion.
		SubmitTo(q, 1, submitted);
		q->Signal(fe.get(), 1);
		HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		fe->SetEventOnCompletion(1, ev);
		const DWORD waited = WaitForSingleObject(ev, 3000);
		CloseHandle(ev);
		int first = -1;
		void* mapped = nullptr;
		const D3D12_RANGE range{ 0, 256ull * 64 };
		if (SUCCEEDED(rb->Map(0, &range, &mapped))) {
			first = static_cast<const uint8_t*>(mapped)[0];
			rb->Unmap(0, nullptr);
		}
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD CROSS TEST ({}): cleared 128, read back {} (fence satisfied={}) "
			"deviceRemovedReason=0x{:08x} -- 128 means this queue still executes the work it is "
			"given",
			what, first, waited == WAIT_OBJECT_0,
			static_cast<uint32_t>(device12->GetDeviceRemovedReason())));
	};
	probeSubmission(queue.get(), "the engine queue the runtime was given");
	probeSubmission(probeQueue.get(), "a queue this backend made for itself");

	// Which of the two it is.
	//
	// The pair at the top of this function is 64x64 in R8G8B8A8 and crosses correctly; the
	// production pair is the frame's own extent in B8G8R8A8 and does not. One more pair at each
	// end of the axis separates the extent from the format, so the fix is aimed at a cause
	// rather than at the shape that happens to exhibit it.
	const auto probePair = [&](uint32_t pw, uint32_t ph, DXGI_FORMAT pf, const char* what) {
		winrt::com_ptr<ID3D11Texture2D> p11;
		winrt::com_ptr<ID3D12Resource> p12;
		if (!CreateSharedTexture(device11, device12.get(), pw, ph, pf, true, p11, p12)) {
			Logger::Get().Info(fmt::format("DLSSNR AMD CROSS TEST ({}): pair not created", what));
			return;
		}
		D3D11_TEXTURE2D_DESC pd{};
		p11->GetDesc(&pd);
		D3D11_TEXTURE2D_DESC st = pd;
		st.Usage = D3D11_USAGE_STAGING;
		st.BindFlags = 0;
		st.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		st.MiscFlags = 0;
		winrt::com_ptr<ID3D11Texture2D> stage;
		if (FAILED(device11->CreateTexture2D(&st, nullptr, stage.put()))) {
			return;
		}
		D3D11_MAPPED_SUBRESOURCE mm{};
		if (FAILED(context11->Map(stage.get(), 0, D3D11_MAP_WRITE, 0, &mm))) {
			return;
		}
		const uint32_t bpp = pf == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 4;
		for (UINT y = 0; y < st.Height; ++y) {
			uint8_t* row = static_cast<uint8_t*>(mm.pData) + size_t(y) * mm.RowPitch;
			for (UINT x = 0; x < st.Width; ++x) {
				row[x * bpp + 0] = 200;
				row[x * bpp + 1] = 200;
				row[x * bpp + 2] = 200;
				row[x * bpp + 3] = 255;
			}
		}
		context11->Unmap(stage.get(), 0);
		context11->CopyResource(p11.get(), stage.get());
		D3D11_QUERY_DESC qd{};
		qd.Query = D3D11_QUERY_EVENT;
		winrt::com_ptr<ID3D11Query> drained;
		if (FAILED(device11->CreateQuery(&qd, drained.put()))) {
			return;
		}
		context11->End(drained.get());
		context11->Flush();
		const ULONGLONG deadline = GetTickCount64() + 3000;
		while (context11->GetData(drained.get(), nullptr, 0, 0) == S_FALSE &&
			GetTickCount64() < deadline) {
			Sleep(0);
		}
		const float r11 = MeanOfOutput(p11.get());
		const float r12 = Measure(p12.get(), pf, D3D12_RESOURCE_STATE_COMMON);
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD CROSS TEST ({}): {}x{} fmt {} -- D3D11 reads {:.4f}, D3D12 reads {:.4f}",
			what, pw, ph, static_cast<uint32_t>(pf), r11, r12));
	};
	probePair(64, 64, DXGI_FORMAT_B8G8R8A8_UNORM, "small extent, production format");
	probePair(width ? width : 3840u, height ? height : 2160u,
		DXGI_FORMAT_R8G8B8A8_UNORM, "production extent, small-pair format");
}

bool DlssnrAmdBackend::Impl::CreatePipeline() noexcept {
	if (root) {
		return true;
	}

	// The layout the engine's shaders are bound against, taken from the working
	// implementation: one SRV from t0, one UAV from u0 at table offset 1, a second table
	// of two SRVs from t1, and twenty-four 32-bit constants. No static sampler. The
	// conversion shaders above use the same layout, which is why there is only one.
	// Two UAVs, not one: the temporal pass writes the residual and the guide it will be
	// measured against next frame. A shader that binds a register the root signature does
	// not cover makes CreateComputePipelineState fail, and that failure used to be silent --
	// see the note in the compile lambda.
	D3D12_DESCRIPTOR_RANGE ranges[2]{};
	ranges[0] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0 };
	ranges[1] = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 1 };

	D3D12_ROOT_PARAMETER params[3]{};
	params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	params[0].DescriptorTable = { 2, ranges };
	params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	params[1].Constants = { 0, 0, 24 };
	params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	// Three, not two: the temporal pass reads a current guide and the guide it was
	// accumulated against, so the second table carries t1 through t3. The conversions and
	// the resolve declare only t1 and t2 and are unaffected.
	// Four, not three: the temporal pass also reads the motion field it reprojects with.
	D3D12_DESCRIPTOR_RANGE residualRange{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 1, 0, 0 };
	params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	params[2].DescriptorTable = { 1, &residualRange };
	params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	D3D12_ROOT_SIGNATURE_DESC desc{ 3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE };
	winrt::com_ptr<ID3DBlob> blob, error;
	if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
		blob.put(), error.put()))) {
		Logger::Get().Error("DLSSNR AMD: root signature serialization failed");
		return false;
	}
	if (FAILED(device12->CreateRootSignature(0, blob->GetBufferPointer(),
		blob->GetBufferSize(), IID_PPV_ARGS(root.put())))) {
		return false;
	}

	D3D12_DESCRIPTOR_HEAP_DESC hd{};
	hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hd.NumDescriptors = 32;
	hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (FAILED(device12->CreateDescriptorHeap(&hd, IID_PPV_ARGS(heap.put())))) {
		return false;
	}
	descriptorStride = device12->GetDescriptorHandleIncrementSize(
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	// The two ends of the conversion, one pipeline each.
	auto compile = [&](const char* source, size_t length, const char* name,
		const D3D_SHADER_MACRO* macros,
		winrt::com_ptr<ID3D12PipelineState>& out) -> bool {
		winrt::com_ptr<ID3DBlob> cs, csError;
		if (FAILED(D3DCompile(source, length, name, macros, nullptr, "main",
			"cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, cs.put(), csError.put()))) {
			Logger::Get().Error(fmt::format("DLSSNR AMD: {} shader failed: {}", name,
				csError ? static_cast<const char*>(csError->GetBufferPointer()) : "?"));
			return false;
		}
		D3D12_COMPUTE_PIPELINE_STATE_DESC ps{};
		ps.pRootSignature = root.get();
		ps.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
		const HRESULT hr = device12->CreateComputePipelineState(&ps,
			IID_PPV_ARGS(out.put()));
		if (FAILED(hr)) {
			// Worth a line of its own. A pipeline whose shader binds a register the root
			// signature does not cover fails here, and this returned false in silence once:
			// the whole backend declined, the effect fell through to a path that does not
			// exist on this machine, and the only evidence was that nothing happened.
			Logger::Get().Error(fmt::format(
				"DLSSNR AMD: {} pipeline failed (0x{:08x}); check the root signature "
				"covers every register it binds", name, static_cast<uint32_t>(hr)));
		}
		return SUCCEEDED(hr);
	};

	// The conversions exist twice: once passing the frame through untouched and once
	// decoding sRGB on the way in and encoding it on the way out. Which one runs is a
	// setting, because whether the engine wants linear input is a question about the
	// picture that only a real frame can answer.
	const D3D_SHADER_MACRO srgbMacros[] = { { "SRGB_IN", "1" }, { nullptr, nullptr } };
	return compile(kConvertInShader, sizeof(kConvertInShader) - 1, "convert in",
			nullptr, convertIn) &&
		compile(kConvertInShader, sizeof(kConvertInShader) - 1, "convert in linear",
			srgbMacros, convertInSrgb) &&
		compile(kConvertOutShader, sizeof(kConvertOutShader) - 1, "convert out",
			nullptr, convertOut) &&
		compile(kConvertOutShader, sizeof(kConvertOutShader) - 1, "convert out linear",
			srgbMacros, convertOutSrgb) &&
		compile(kDownsampleShader, sizeof(kDownsampleShader) - 1, "downsample",
			nullptr, downsample) &&
		compile(kResolveShader, sizeof(kResolveShader) - 1, "resolve",
			nullptr, resolve) &&
		compile(kTemporalShader, sizeof(kTemporalShader) - 1, "temporal",
			nullptr, temporal) &&
		compile(kMotionShader, sizeof(kMotionShader) - 1, "motion resample",
			nullptr, motionResample);
}

bool DlssnrAmdBackend::Impl::CreateSized(
	uint32_t w, uint32_t h, DXGI_FORMAT inFmt, DXGI_FORMAT outFmt
) noexcept {
	DestroySized();
	width = w;
	height = h;
	inputFormat = inFmt;
	outputFormat = outFmt;
	inputIsFp16 = inFmt == DXGI_FORMAT_R16G16B16A16_FLOAT;
	outputIsFp16 = outFmt == DXGI_FORMAT_R16G16B16A16_FLOAT;

	// Each shared texture carries the format its own side of Magpie uses. The engine's
	// surfaces are always RGBA16F, so a conversion sits at whichever end differs --
	// often both. The channel order those eight-bit formats name is part of the format,
	// not something the conversion has to fix up, so no conversion swaps anything.
	// The shader that fills the shared input, and the view it writes through. Both are built
	// once here, since neither depends on the frame.
	{
		winrt::com_ptr<ID3DBlob> blob;
		if (!Magpie::DirectXHelper::CompileComputeShader(
				kFillSharedShader, "main", blob.put(), "DLSSNRFillShared")) {
			Logger::Get().Error("DLSSNR AMD: the shared-input fill shader failed");
			return false;
		}
		if (FAILED(device11->CreateComputeShader(blob->GetBufferPointer(),
				blob->GetBufferSize(), nullptr, fillSharedShader.put()))) {
			Logger::Get().Error("DLSSNR AMD: could not create the fill shader");
			return false;
		}
	}

	// Before the runtime is loaded. It hooks DXGI and the D3D12 command queue when it arrives,
	// and if the crossing works here and not afterwards, that hook is the whole difference.
	CrossTest();

	// The fence the two devices agree on, before either is used. It is created on D3D11 and
	// opened on D3D12, the same way the textures are, so both sides are talking about the same
	// object rather than each about its own idea of where the other has got to.
	{
		// One fence per direction, both made the same way: created on D3D11 and opened on
		// D3D12 so the two sides are talking about the same object rather than each about its
		// own idea of where the other has got to.
		const auto makeCrossingFence = [&](const char* what,
				winrt::com_ptr<ID3D11Fence>& fence11,
				winrt::com_ptr<ID3D12Fence>& fence12) {
			if (FAILED(device11->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
					IID_PPV_ARGS(fence11.put())))) {
				Logger::Get().Error(fmt::format(
					"DLSSNR AMD: could not create the {} fence", what));
				return false;
			}
			HANDLE raw = nullptr;
			if (FAILED(fence11->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &raw))) {
				Logger::Get().Error(fmt::format(
					"DLSSNR AMD: could not share the {} fence", what));
				return false;
			}
			wil::unique_handle handle(raw);
			if (FAILED(device12->OpenSharedHandle(handle.get(),
					IID_PPV_ARGS(fence12.put())))) {
				Logger::Get().Error(fmt::format(
					"DLSSNR AMD: could not open the {} fence", what));
				return false;
			}
			return true;
		};
		if (!makeCrossingFence("input crossing", inFence11, inFence12) ||
			!makeCrossingFence("output crossing", outFence11, outFence12)) {
			return false;
		}
		if (FAILED(context11->QueryInterface(IID_PPV_ARGS(context11x.put())))) {
			Logger::Get().Error("DLSSNR AMD: the renderer's context cannot wait on a fence");
			return false;
		}
	}

	// allowUav on the input pair, though this backend writes it with CopyResource and never
	// binds a UAV: both implementations in this codebase that cross this boundary successfully
	// -- the AMD optical flow provider and the older DLSSNR filter -- declare it that way, and
	// neither uses a plain copy. This is the smallest step that matches them before rewriting
	// the fill path itself.
	if (!CreateSharedTexture(device11, device12.get(), w, h, inFmt, true,
			sharedIn11, sharedIn12) ||
		!CreateSharedTexture(device11, device12.get(), w, h, outFmt, true,
			sharedOut11, sharedOut12)) {
		Logger::Get().Error("DLSSNR AMD: could not share the frame between the devices");
		return false;
	}
	if (FAILED(device11->CreateUnorderedAccessView(sharedIn11.get(), nullptr,
			sharedInUav.put()))) {
		Logger::Get().Error("DLSSNR AMD: could not make the shared input writable");
		return false;
	}

	// The network's extent. Everything the engine touches is built at this size, and at
	// 100% it is the capture size, which is the path that shipped before the scale existed.
	//
	// The range is the reference's own: it clamps its model scale to a quarter at the least.
	// This backend held the short side at 540 pixels for a while, which is stricter than the
	// reference and was justified by two things that are both gone. One was a probe reading
	// showing the engine's output collapsing to a sixteenth of its input at 480x270 -- that
	// figure came from a readback sized for the capture rather than for the surface being
	// read, so it measured uninitialised memory and meant nothing. The other was the engine's
	// exposure hunting, which stopped when the host began supplying an exposure. Left in
	// place it silently raised every request below half, which is what the user saw: 25%
	// asked for, 50% applied, and a frame rate to match.
	const uint32_t scaledW = uint32_t(float(w) * modelScale + 0.5f);
	const uint32_t scaledH = uint32_t(float(h) * modelScale + 0.5f);
	netWidth = std::clamp(scaledW, 32u, w);
	netHeight = std::clamp(scaledH, 32u, h);

	downscaling = netWidth != w || netHeight != h;
	// The resolve path runs whenever the network is smaller than the capture, and also when
	// anti-flicker is on: the residual has to be extracted and composited somewhere, and that
	// is what the resolve does -- at 1:1 when nothing is scaled.
	scaled = downscaling || antiFlickerMode != 0;

	// The frame at full resolution, in the engine's own format. It is what the capture is
	// converted into, what the resolve takes its detail from, and at 100% scale it is also
	// what the network is handed.
	for (uint32_t i = 0; i < kSlots; ++i) {
		if (!CreateEngineTexture(device12.get(), w, h,
				DXGI_FORMAT_R16G16B16A16_FLOAT, full[i])) {
			return false;
		}
	}

	// The engine's surface. It works in place: the texture named in the packet is both what
	// it reads and what it writes, so there is one surface there, not two. The reference
	// hands over exactly one such pointer as `Packet::colour`.
	for (uint32_t i = 0; i < kSlots; ++i) {
		if (!CreateEngineTexture(device12.get(), netWidth, netHeight,
				DXGI_FORMAT_R16G16B16A16_FLOAT, net[i])) {
			return false;
		}
	}
	if (scaled) {
		// The network's output for this frame before the engine ran, and where the resolve
		// writes. The resolve subtracts the first from what the engine produced to isolate
		// the edit, and needs `full` to put the untouched detail back underneath it.
		for (uint32_t i = 0; i < kSlots; ++i) {
			if (!CreateEngineTexture(device12.get(), netWidth, netHeight,
					DXGI_FORMAT_R16G16B16A16_FLOAT, baseline[i])) {
				return false;
			}
		}
		if (!CreateEngineTexture(device12.get(), w, h,
				DXGI_FORMAT_R16G16B16A16_FLOAT, resolved)) {
			return false;
		}
		// The residual history, at the resolution the network ran at, which is where the
		// edit lives. Two of each, because the pass reads one pair and writes the other.
		for (int i = 0; i < 2; ++i) {
			if (!CreateEngineTexture(device12.get(), netWidth, netHeight,
					DXGI_FORMAT_R16G16B16A16_FLOAT, historyResidual[i]) ||
				!CreateEngineTexture(device12.get(), netWidth, netHeight,
					DXGI_FORMAT_R16G16B16A16_FLOAT, historyGuide[i])) {
				return false;
			}
		}
		historyIndex = 0;
		temporalValid = false;
	}
	// The engine's input contract names motion and depth. This backend has no source for
	// either, and zero-filled buffers are what the reference feeds when a game offers
	// neither; depth stays disabled on the engine side to match. They follow the network's
	// extent rather than the capture's.
	for (uint32_t i = 0; i < kSlots; ++i) {
		if (!CreateEngineTexture(device12.get(), netWidth, netHeight,
				DXGI_FORMAT_R16G16_FLOAT, motion[i]) ||
			!CreateEngineTexture(device12.get(), netWidth, netHeight,
				DXGI_FORMAT_R32_FLOAT, depth[i])) {
			return false;
		}
	}
	// Held steady rather than adapted; the reason and the measurements are on kExposureValue.
	if (!CreateExposure()) {
		return false;
	}
	return true;
}

bool DlssnrAmdBackend::Impl::CreateReconstruction(
	DeviceResources& resources, ID3D11Texture2D* output
) noexcept {
	reconstruction.reset();
	netShared11 = nullptr;
	netShared12 = nullptr;
	if (!reconstruct || !netWidth || !netHeight || !output) {
		return true;
	}
#ifndef MP_ENABLE_FSR3_ZEROMV
	Logger::Get().Warn(
		"DLSSNR AMD: the reconstruction route was asked for but this build has no FSR3 "
		"upscaler; falling back to the resolve");
	return true;
#else
	// The placeholders: two that are never read, one that stands in for motion until the
	// provider has produced any. None of them needs contents.
	auto makePlain = [this](DXGI_FORMAT format, winrt::com_ptr<ID3D11Texture2D>& out) noexcept {
		D3D11_TEXTURE2D_DESC desc{};
		desc.Width = netWidth;
		desc.Height = netHeight;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.Format = format;
		desc.SampleDesc.Count = 1;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		return SUCCEEDED(device11->CreateTexture2D(&desc, nullptr, out.put()));
	};
	// The colour the upscaler reads: written here, read by the other device -- the direction
	// sharedOut has always used. It carries the shoulder and the sRGB encode, applied before
	// the upscaler rather than after it, for two reasons: what it reconstructs from is then
	// display-referred, which is what it expects by default, and the result needs no journey
	// back, because the upscaler writes the display texture itself -- which is what the first
	// version of this route did, and the only version that ever ran clean.
	//
	// Everything this replaces came from reading its output back on this queue: a surface
	// written by one device and read by the other in that direction, which is the one crossing
	// this backend has never done and which took the device down every time.
	if (!CreateSharedTexture(device11, device12.get(), netWidth, netHeight,
			DXGI_FORMAT_R16G16B16A16_FLOAT, false, netShared11, netShared12)) {
		Logger::Get().Error("DLSSNR AMD: the reconstruction could not share its colour");
		return false;
	}
	if (!CreateEngineTexture(device12.get(), netWidth, netHeight,
			DXGI_FORMAT_R16G16B16A16_FLOAT, encoded)) {
		Logger::Get().Error("DLSSNR AMD: the encode could not get a surface of its own");
		return false;
	}
	if (!CreateSharedTexture(device11, device12.get(), netWidth, netHeight,
			DXGI_FORMAT_R16G16_FLOAT, false, motionShared11, motionShared12)) {
		Logger::Get().Error("DLSSNR AMD: the reconstruction could not share its motion");
		return false;
	}
	if (!makePlain(DXGI_FORMAT_R16G16_FLOAT, zeroMotion11) ||
		!makePlain(DXGI_FORMAT_R32_FLOAT, guideDepth11) ||
		!makePlain(DXGI_FORMAT_R8_UNORM, guideConfidence11)) {
		Logger::Get().Error("DLSSNR AMD: the reconstruction could not build its guide");
		return false;
	}
	if (FAILED(device11->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
			IID_PPV_ARGS(crossFence11.put())))) {
		Logger::Get().Error("DLSSNR AMD: the crossing could not create its fence");
		return false;
	}
	HANDLE rawCross = nullptr;
	if (FAILED(crossFence11->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &rawCross))) {
		Logger::Get().Error("DLSSNR AMD: the crossing could not share its fence");
		return false;
	}
	{
		wil::unique_handle crossHandle(rawCross);
		if (FAILED(device12->OpenSharedHandle(crossHandle.get(),
				IID_PPV_ARGS(crossFence12.put())))) {
			Logger::Get().Error("DLSSNR AMD: the crossing could not open its fence");
			return false;
		}
	}
	// Version 3.1.5, deliberately: FSR 4.1.1 is the FSR4\FSR4_SR effect's own business, and
	// that effect already asks for it. This route is the reconstruction the DLSSNR network
	// feeds, not a super-resolution mode of its own.
	auto upscaler = std::make_unique<FSR3Upscaler>();
	if (!upscaler->Initialize(resources, netShared11.get(), output,
			settings.motionRequest)) {
		Logger::Get().Error("DLSSNR AMD: the FSR3 upscaler refused to initialise");
		return false;
	}
	// The protocol the renderer would have set if this upscaler were a link in the chain.
	// What it is handed carries the shoulder and the sRGB encode, applied before it rather
	// than after, so it is told it is reading display-referred colour: hdrColorInput stays
	// false, which is also the default it would have had. The stage that used to run after it
	// is gone, and with it the only reason this flag ever needed to be true.
	upscaler->SetFsrHdrProtocol(FsrHdrProtocol{
		.hdrColorInput = false,
		.transfer = GroupBTransfer::Linear,
		.preExposure = 1.0f,
		.exposure = 1.0f,
		.depthInverted = true,
		.depthInfinite = true,
		.useReactiveMask = false,
		.useTransparencyMask = false,
	});

	reconstruction = std::move(upscaler);
	// Said out loud because the sizes are the whole point of the route: the engine edits at
	// netWidth by netHeight and the accumulator reconstructs the captured size from it.
	Logger::Get().Info(fmt::format(
		"DLSSNR AMD: reconstruction on -- FSR3 upscaler from {}x{} to the captured size, "
		"motion {}, colour shoulder + sRGB encoded before it, output written by it",
		netWidth, netHeight,
		settings.motionRequest.method == OpticalFlowMethod::None ? "none (zero-MV route)"
			: "from the frame guidance"));
	return true;
#endif
}

bool DlssnrAmdBackend::Impl::RunReconstruction(
	const NativeEffectDrawContext& context
) noexcept {
#ifdef MP_ENABLE_FSR3_ZEROMV
	// The engine's submission is on the queue whether or not it answered, so the allocator
	// cannot be handed to a new list until it has run -- the same wait the composite path
	// does before it records anything.
	// Each exit names itself. The first version reported them all as one message, and that
	// cost a round trip: the route failed, the log said only that it had, and the answer was
	// a contract check two layers down inside the upscaler.
	const auto fail = [this](const std::string& where) noexcept {
		DumpDeviceMessages(where.c_str());
		if (++reconstructionFailures <= 6) {
			Logger::Get().Warn(fmt::format(
				"DLSSNR AMD: the reconstruction stopped at {}", where));
		}
		return false;
	};
	const uint64_t submitted = ++fenceValue;
	queue->Signal(fence.get(), submitted);
	if (fence->GetCompletedValue() < submitted) {
		fence->SetEventOnCompletion(submitted, fenceEvent.get());
		if (WaitForSingleObject(fenceEvent.get(), 1000) != WAIT_OBJECT_0) {
			return fail("waiting for the engine's submission");
		}
	}
	if (FAILED(allocator->Reset()) ||
		FAILED(list->Reset(allocator.get(), nullptr))) {
		return fail("resetting the allocator");
	}
	// The engine's frame through the same shoulder and encode the output stage applies -- run
	// here, at the network's resolution, so the upscaler reconstructs from display-referred
	// colour.
	// Recorded in as many steps as the settings file asks for. Stopping between them is the
	// only way this was narrowed down: every step is a call the working path also makes, and
	// only one of them takes the device with it.
	Barrier(list.get(), encoded.get(), kStateShaderRead,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	if (encodeSteps >= 2) {
		Bind(6, net[slot].get(), DXGI_FORMAT_R16G16B16A16_FLOAT, encoded.get(),
			DXGI_FORMAT_R16G16B16A16_FLOAT);
	}
	if (encodeSteps >= 4) {
		// The shoulder travels as the dispatch's own constants, not as a root value recorded
		// before it. That distinction is the whole of why this route kept removing the device:
		// a root value written to a list that has no root signature yet is written to nothing,
		// and the driver takes the device away for it. The pipeline and the root signature are
		// bound inside the dispatch, so anything the shader reads has to be handed over there.
		//
		// Sized to the network as well, not to the capture: this pass runs on the surface it
		// writes here, which is the network's extent, and the plain Dispatch would have
		// launched it over the whole frame.
		const UINT encodeSettings[1]{ shoulderMilli };
		DispatchSized(srgbInput ? convertOutSrgb.get() : convertOut.get(), 6,
			netWidth, netHeight, 0, encodeSettings, 1);
	}
	Barrier(list.get(), encoded.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		kStateShaderRead);
	if (encodeSteps >= 5) {
		// The crossing, as a copy and nothing more.
		Barrier(list.get(), encoded.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(list.get(), netShared12.get(), D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_COPY_DEST);
		list->CopyResource(netShared12.get(), encoded.get());
		Barrier(list.get(), netShared12.get(), D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_COMMON);
		Barrier(list.get(), encoded.get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
			kStateShaderRead);
	}
	// Said once a frame in case the debug layer is on: it reports what the call above did
	// wrong here rather than at the point the device finally goes.
	if (reconstructionDraws == 0) {
		DumpDeviceMessages("encode steps");
	}
	// Submitted on its own. The route failed at Close() with no hint of which of the calls
	// above it disliked -- Close reports what the list recorded, and a list that has recorded
	// several things says only that one of them was wrong. Split here, the message names the
	// half.
	const HRESULT encodeResult = list->Close();
	if (FAILED(encodeResult)) {
		return fail(fmt::format("closing the encode list (0x{:08x})",
			static_cast<uint32_t>(encodeResult)));
	}
	ID3D12CommandList* encodeLists[] = { list.get() };
	SubmitTo(queue.get(), 1, encodeLists);
	const uint64_t encodeDone = ++fenceValue;
	queue->Signal(fence.get(), encodeDone);
	if (fence->GetCompletedValue() < encodeDone) {
		fence->SetEventOnCompletion(encodeDone, fenceEvent.get());
		if (WaitForSingleObject(fenceEvent.get(), 1000) != WAIT_OBJECT_0) {
			return fail("waiting for the encode");
		}
	}
	if (FAILED(allocator->Reset()) ||
		FAILED(list->Reset(allocator.get(), nullptr))) {
		return fail("resetting the allocator for the motion");
	}
	// The motion this backend resampled to the network's extent, on the same submission, so
	// the single fence below covers both and the upscaler reads a finished pair.
	if (motionReady) {
		Barrier(list.get(), motionShared12.get(), D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_COPY_DEST);
		list->CopyResource(motionShared12.get(), motion[slot].get());
		Barrier(list.get(), motionShared12.get(), D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_COMMON);
	}
	const HRESULT copiedList = list->Close();
	if (FAILED(copiedList)) {
		return fail(fmt::format("closing the copy list (0x{:08x})",
			static_cast<uint32_t>(copiedList)));
	}
	ID3D12CommandList* lists[] = { list.get() };
	SubmitTo(queue.get(), 1, lists);
	const uint64_t copied = ++fenceValue;
	queue->Signal(fence.get(), copied);
	if (fence->GetCompletedValue() < copied) {
		fence->SetEventOnCompletion(copied, fenceEvent.get());
		if (WaitForSingleObject(fenceEvent.get(), 1000) != WAIT_OBJECT_0) {
			return fail("waiting for the copy");
		}
	}

	// The handshake, before anything reads what was just written. The queue has finished, which
	// says the writes are done; it does not say the other device can see them. A shared surface
	// needs the two sides to agree on the point in the stream where one's writes become the
	// other's reads, and that agreement is this fence.
	{
		const uint64_t crossing = ++crossValue;
		queue->Signal(crossFence12.get(), crossing);
		if (FAILED(context11->Wait(crossFence11.get(), crossing))) {
			return fail("the other device refusing the crossing");
		}
	}

	// A guidance view at the upscaler's own extent, built rather than inherited: see the note
	// on the motion surfaces. The sync point is left empty because the copy above was ordered
	// by this queue and the fence before it, so there is nothing else to wait on.
	const FrameGuidanceExtent extent{ netWidth, netHeight };
	FrameGuidanceMetadata meta{};
	meta.frameId = context.frameId;
	meta.sourceExtent = extent;
	meta.validRegion = FrameGuidanceRegion{ 0, 0, netWidth, netHeight };
	meta.valid = true;
	FrameGuidanceView guidance{};
	guidance.motion = { motionReady ? motionShared11.get() : zeroMotion11.get(),
		DXGI_FORMAT_R16G16_FLOAT, meta };
	guidance.depth = { guideDepth11.get(), DXGI_FORMAT_R32_FLOAT, meta };
	guidance.confidence = { guideConfidence11.get(), DXGI_FORMAT_R8_UNORM, meta };
	guidance.requiresHistoryReset = context.frameGuidance.requiresHistoryReset;

	// The phase the downsample was given, passed on so the upscaler can put this frame back on
	// the grid it came from. Without it every frame looks like the same sample of the same
	// place and accumulates into nothing.
	reconstruction->SetJitter(frameJitterX, frameJitterY);

	// The upscaler does its own copying, its own fences and its own barriers from here: it is
	// handed the shared texture, not a command list.
	const NativeEffectDrawContext hand{
		.input = netShared11.get(),
		.output = context.output,
		.frameId = context.frameId,
		.inputRevision = context.inputRevision,
		.frameGuidance = guidance,
		.zeroFrameGuidance = guidance,
	};
	// Checked here as well as inside the upscaler, so a rejection says which of the two it
	// was: a view this side built wrong, or something further in.
	if (!guidance.IsValidFor(context.frameId, extent)) {
		return fail("its own guide failing the upscaler's contract");
	}
	if (!reconstruction->Draw(hand)) {
		return fail("the upscaler returning false with a valid guide");
	}
	// Nothing to do afterwards. The upscaler wrote the display texture itself, which is what it
	// is built to do -- in the chain its output is the next link's input, and here it is the
	// frame.
	//
	// The reading is taken here, after both submissions, because measuring resets the
	// allocator and re-records the list. Placed any earlier it discards the recording that is
	// still open -- which is not a small mistake: it silently throws the encode away and the
	// upscaler then reconstructs from a frame nothing wrote to.
	// Not the first frame: the upscaler starts its history there and a sequence has to build
	// before there is anything to measure. The hundred-and-twentieth is settled.
	if (reconstructionDraws == 120) {
		// Three points on one frame: what the network produced, what the upscaler was handed,
		// and what came out. The encode is meant to lift the first into the second, and the
		// third should land near the second -- a picture much darker than its own input is the
		// symptom this ordering was chosen to avoid.
		const float netMean = Measure(net[slot].get(), DXGI_FORMAT_R16G16B16A16_FLOAT,
			kStateShaderRead);
		const float handedMean = encodeSteps >= 5
			? Measure(netShared12.get(), DXGI_FORMAT_R16G16B16A16_FLOAT,
				D3D12_RESOURCE_STATE_COMMON)
			: -1.0f;
		float detail = -1.0f;
		const float onScreen = MeanOfOutput(context.output, &detail);
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD: settled reconstruction (frame {}) -- network {:.4f}, handed over "
			"{:.4f}, on screen {:.4f}, detail {:.5f} (jitter {})",
			reconstructionDraws,
			netMean, handedMean, onScreen, detail, jitterEnabled ? "on" : "off"));
	}
	++reconstructionDraws;
	if (reconstructionDraws == 1 || reconstructionDraws % 300 == 0) {
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD: reconstruction frame {} (motion {})", reconstructionDraws,
			motionReady ? "from this backend's resample" : "not ready, zero"));
	}
	return true;
#else
	(void)context;
	return false;
#endif
}

// Tells the runtime to stop referring to the history it was carrying.
//
// The runtime holds a pointer to the last history it was given, and that storage is this
// backend's and is released with the session. Clearing it used to happen lazily, from the first
// SubmitEngineJob after the fact -- which leaves the runtime describing freed memory for as long
// as it takes the next session to submit anything, and whatever drives it in that window (its
// own worker finishing a job, a present, a frame generator's proxy Present) is reading and
// writing storage that has already been handed back.
//
// That is the shape of the 20:00:53 run: session two set up cleanly, ran about twenty-five jobs,
// and then the runtime's own HIP kernel faulted -- `device sync 719 (unspecified launch
// failure)`, and the runtime's log says so itself. A kernel fault is what a stale pointer looks
// like when the storage behind it has been recycled.
//
// Called where the storage goes away, so there is no window to be in.
void DlssnrAmdBackend::Impl::ForgetHistory() noexcept {
	if (!runtime || !rva) {
		return;
	}
	At<uint8_t>(runtime, rva->wantHistory) = 0;
	At<void*>(runtime, rva->history) = nullptr;
	temporalValid = false;
}

void DlssnrAmdBackend::Impl::DestroySized() noexcept {
	// Before the surfaces: the upscaler holds the shared texture it was given, and it has to
	// let go of one that is about to be replaced.
	reconstruction.reset();
	netShared11 = nullptr;
	netShared12 = nullptr;
	encoded = nullptr;
	crossFence11 = nullptr;
	crossFence12 = nullptr;
	crossValue = 0;

	motionShared11 = nullptr;
	motionShared12 = nullptr;
	zeroMotion11 = nullptr;
	guideDepth11 = nullptr;
	guideConfidence11 = nullptr;
	sharedIn11 = nullptr;
	sharedIn12 = nullptr;
	sharedOut11 = nullptr;
	sharedOut12 = nullptr;
	inFence11 = nullptr;
	inFence12 = nullptr;
	inFenceValue = 0;
	outFence11 = nullptr;
	outFence12 = nullptr;
	outFenceValue = 0;
	for (uint32_t i = 0; i < kSlots; ++i) {
		net[i] = nullptr;
		full[i] = nullptr;
		baseline[i] = nullptr;
		motion[i] = nullptr;
		depth[i] = nullptr;
	}
	slot = 0;
	for (uint32_t i = 0; i < kSlots; ++i) {
		slotJob[i] = 0;
		slotFence[i] = 0;
	}
	outFence = 0;
	resolved = nullptr;
	// Before the storage is handed back, not after the next session has already started using it.
	ForgetHistory();
	for (int i = 0; i < 2; ++i) {
		historyResidual[i] = nullptr;
		historyGuide[i] = nullptr;
	}
	// A new extent means the history describes the wrong geometry.
	resetHistory = true;
	temporalValid = false;
}

void DlssnrAmdBackend::Impl::CreateSrv(ID3D12Resource* res,
	D3D12_CPU_DESCRIPTOR_HANDLE where
) noexcept {
	D3D12_SHADER_RESOURCE_VIEW_DESC s{};
	s.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
	s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	s.Texture2D.MipLevels = 1;
	device12->CreateShaderResourceView(res, &s, where);
}

void DlssnrAmdBackend::Impl::Bind(
	uint32_t tableSlot, ID3D12Resource* srv, DXGI_FORMAT srvFormat,
	ID3D12Resource* uav, DXGI_FORMAT uavFormat
) noexcept {
	D3D12_SHADER_RESOURCE_VIEW_DESC s{};
	s.Format = srvFormat;
	s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	s.Texture2D.MipLevels = 1;
	device12->CreateShaderResourceView(srv, &s, Cpu(tableSlot));

	D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
	u.Format = uavFormat;
	u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device12->CreateUnorderedAccessView(uav, nullptr, &u, Cpu(tableSlot + 1));
}

void DlssnrAmdBackend::Impl::Dispatch(ID3D12PipelineState* pso, uint32_t tableSlot) noexcept {
	DispatchSized(pso, tableSlot, width, height, false);
}

void DlssnrAmdBackend::Impl::DispatchSized(ID3D12PipelineState* pso, uint32_t tableSlot,
	uint32_t dw, uint32_t dh, uint32_t residualTableSlot, const UINT* constants,
	uint32_t constantCount
) noexcept {
	// `tableSlot` matters more than it looks. Descriptors are read when the GPU executes the
	// list, not when it is recorded, so two stages sharing a tableSlot both end up reading
	// whichever bind happened last. Each stage therefore gets its own set, and the root
	// table is pointed at that set here.
	ID3D12DescriptorHeap* h = heap.get();
	list->SetComputeRootSignature(root.get());
	list->SetDescriptorHeaps(1, &h);
	D3D12_GPU_DESCRIPTOR_HANDLE table = heap->GetGPUDescriptorHandleForHeapStart();
	table.ptr += UINT64(tableSlot) * descriptorStride;
	list->SetComputeRootDescriptorTable(0, table);
	if (residualTableSlot != 0) {
		// The extra sources live in a table of their own; where it starts depends on how many
		// descriptors the pass put before it.
		D3D12_GPU_DESCRIPTOR_HANDLE residual = heap->GetGPUDescriptorHandleForHeapStart();
		residual.ptr += UINT64(residualTableSlot) * descriptorStride;
		list->SetComputeRootDescriptorTable(2, residual);
	}
	list->SetPipelineState(pso);
	// The downsample and the resolve are the only stages with an extent to pass, and they
	// take it in the same four-tableSlot shape the reference uses: destination width and height
	// followed by the source's.
	if (constants) {
		list->SetComputeRoot32BitConstants(1, constantCount, constants, 0);
	}
	list->Dispatch((dw + 7) / 8, (dh + 7) / 8, 1);
}

void DlssnrAmdBackend::Impl::BindResolve(uint32_t tableSlot, ID3D12Resource* srv,
	ID3D12Resource* uav, ID3D12Resource* baselineTexture, ID3D12Resource* edited
) noexcept {
	// t0/u0 through the usual pair, then t1 and t2 sitting immediately after it so the
	// resolve's second root table finds them two descriptors along.
	Bind(tableSlot, srv, DXGI_FORMAT_R16G16B16A16_FLOAT, uav, DXGI_FORMAT_R16G16B16A16_FLOAT);

	CreateSrv(baselineTexture, Cpu(tableSlot + 2));
	CreateSrv(edited, Cpu(tableSlot + 3));
}

void DlssnrAmdBackend::Impl::BindTemporal(uint32_t tableSlot, ID3D12Resource* edited,
	ID3D12Resource* nextResidual, ID3D12Resource* nextGuide,
	ID3D12Resource* baselineTexture, ID3D12Resource* historyResidualTexture,
	ID3D12Resource* historyGuideTexture, ID3D12Resource* motionTexture
) noexcept {
	// t0 and u0 through the usual pair, then u1 immediately after it, and the four sources
	// from t1 in the table that follows -- which is why this pass's second table starts at
	// tableSlot + 3 while the resolve's starts at tableSlot + 2.
	Bind(tableSlot, edited, DXGI_FORMAT_R16G16B16A16_FLOAT, nextResidual,
		DXGI_FORMAT_R16G16B16A16_FLOAT);

	D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
	u.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
	u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device12->CreateUnorderedAccessView(nextGuide, nullptr, &u, Cpu(tableSlot + 2));

	CreateSrv(baselineTexture, Cpu(tableSlot + 3));
	CreateSrv(historyResidualTexture, Cpu(tableSlot + 4));
	CreateSrv(historyGuideTexture, Cpu(tableSlot + 5));

	D3D12_SHADER_RESOURCE_VIEW_DESC m{};
	m.Format = DXGI_FORMAT_R16G16_FLOAT;
	m.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	m.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	m.Texture2D.MipLevels = 1;
	device12->CreateShaderResourceView(motionTexture, &m, Cpu(tableSlot + 6));
}

// The residual for this frame: what the network changed, at the resolution it ran at, with
// earlier frames carried into it where the effect asked for them. Always runs on the
// resolve path; with anti-flicker off it is a subtraction and nothing more.
bool DlssnrAmdBackend::Impl::RunTemporal(const NativeEffectDrawContext& context) noexcept {
	const uint32_t from = historyIndex;
	const uint32_t to = 1 - historyIndex;

	uint32_t weightMilli = 0;
	const bool hasHistory = antiFlickerMode != 0 && temporalValid;
	const uint64_t now = GetTickCount64();
	const bool duplicate =
		context.frameId == lastFrameId && context.inputRevision == lastRevision;
	if (hasHistory && !duplicate && context.frameId > lastFrameId &&
		context.inputRevision == lastRevision && now > lastTimestamp) {
		const double seconds = double(now - lastTimestamp) * 1e-3;
		if (seconds <= 0.25) {
			weightMilli = static_cast<uint32_t>(
				std::clamp(std::exp(-seconds / 0.08), 0.0, 1.0) * 1000.0);
		}
	}

	Barrier(list.get(), historyResidual[to].get(), kStateShaderRead,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	Barrier(list.get(), historyGuide[to].get(), kStateShaderRead,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	// The same generation the picture comes from: this pass reprojects a history against the
	// frame the engine has finished, and pairing it with the frame still being recorded would
	// align the past against something that does not exist yet.
	const uint32_t sourceSlot = pictureSlot;
	BindTemporal(12, net[sourceSlot].get(), historyResidual[to].get(), historyGuide[to].get(),
		baseline[sourceSlot].get(), historyResidual[from].get(), historyGuide[from].get(),
		motion[sourceSlot].get());
	// Modes two and above reproject the history with the flow; one is static accumulation.
	const uint32_t useMotion =
		antiFlickerMode >= 2 && motionReady ? 1u : 0u;
	// The residual controls travel as the floats they are: the shader declares them float,
	// so the bit pattern has to be forwarded rather than converted, which is what BitsOfFloat
	// does and what every other float-carrying dispatch here already uses.
	const UINT constants[12]{ netWidth, netHeight, weightMilli, hasHistory ? 1u : 0u,
		useMotion,
		uint32_t(motionScaleX * 1000.0f), uint32_t(motionScaleY * 1000.0f),
		BitsOfFloat(settings.residualMultiplier), BitsOfFloat(settings.residualSaturation),
		BitsOfFloat(settings.residualLightness),
		BitsOfFloat(settings.shadowStructureMultiplier),
		BitsOfFloat(settings.reflectionGlowMultiplier) };
	DispatchSized(temporal.get(), 12, netWidth, netHeight, 15, constants, 12);
	Barrier(list.get(), historyResidual[to].get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		kStateShaderRead);
	Barrier(list.get(), historyGuide[to].get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		kStateShaderRead);

	historyIndex = to;
	temporalValid = true;
	lastFrameId = context.frameId;
	lastRevision = context.inputRevision;
	lastTimestamp = now;
	return true;
}

bool DlssnrAmdBackend::Impl::InitEngine(const std::filesystem::path& weightsPath) noexcept {
	// Initialise the engine once for the life of the process.
	//
	// The other three steps of the bring-up are already idempotent -- LoadRuntime returns
	// early when `runtime` is set, CreateHip when `hipSet`, CreatePipeline when `root` -- and
	// this one was not, so turning the effect off and on again called the runtime's init a
	// second time while the first engine was still live: its worker threads still running,
	// its device, queue and history still registered, because Impl's destructor deliberately
	// unloads nothing. The second call is what crashed: the log from such a run ends with
	//
	//     its capture never landed (captured word 0): the game did not execute the command
	//     list our capture was recorded into
	//     FAULT: exception 0xc0000005 at amdhip64_7.dll + 0x3f496f
	//
	// followed by a driver reset. That is the queue the engine had already rebuilt being
	// rebuilt again underneath it.
	//
	// Reusing the live engine is the right answer rather than resetting and re-initialising:
	// the weights are the same, the device is Magpie's process-wide one, and the note below
	// records that the engine holds device references for the life of the process -- it is
	// not something this backend can take down and stand back up. Nothing it needs changes
	// between one session and the next except the frame-sized surfaces, and those belong to
	// CreateSized, which does run again.
	if (engineUp) {
		return true;
	}

	// The engine is process state, so a later session adopts it instead of initialising it
	// again, and this is the branch that matters: measured, a second init in one process ran
	// `engine bring-up 2` on a live engine and faulted the GPU within six seconds, while the
	// same process's `engine bring-up 1` carried 36 seconds of frames without a fault.
	//
	// Everything below -- the adapter check, the field writes, the init call itself -- is the
	// first session's work and must not happen twice.
	//
	// What adopting costs: the engine keeps the staging it built at that first init, and the
	// guide flags below are read at staging time. Changing one of those needs Magpie restarted
	// rather than the effect switched off and on. That is a real limitation, and it is worth
	// far more than the bugcheck it replaces.
	if (Engine().started) {
		Logger::Get().Info(
			"DLSSNR AMD: the engine is already up in this process; adopting it rather than "
			"initialising a second time");
		engineUp = true;
		return true;
	}

	// Not captured here. The entry saved when the queue was created is the driver's; the one
	// standing now is the runtime's hook, because its DLL was loaded before this function ran.
	// Reported so the two can be told apart, since every submission this backend makes goes
	// through the saved one.
	{
		const auto current = reinterpret_cast<uintptr_t>(
			(*reinterpret_cast<void***>(queue.get()))[10]);
		const auto saved = reinterpret_cast<uintptr_t>(executeOriginal);
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD: the queue's vtable entry is {:x} now; {:x} was saved before the runtime "
			"loaded; {}",
			current, saved, current == saved ? "same -- nothing hooked it" :
				"different -- the runtime hooked it, and submissions use the saved one"));
	}
	if (!executeOriginal) {
		Logger::Get().Error("DLSSNR AMD: the queue has no ExecuteCommandLists to save");
		return false;
	}

	// The engine holds these for the life of the process, so they are referenced rather than
	// merely borrowed: the backend's own pointers are released when a session ends, and the
	// engine is not part of that.
	device12.get()->AddRef();
	queue.get()->AddRef();
	// Deliberately NOT written, and this is the finding of the whole investigation.
	//
	// The trampoline slot holds a pointer the runtime calls from notify, and the layout notes say
	// to point it at a no-op because the host has already submitted. Writing that no-op here is
	// what stops this process's D3D12 from executing anything at all: a crossing test placed
	// either side of this one statement passes on its way in and fails on its way out, on both
	// the engine's queue and one this backend made, through the driver's own
	// ExecuteCommandLists, with the device healthy and its fences still advancing. Everything
	// measured downstream of that write reads zero, which is the black screen.
	//
	// Left alone, the slot keeps whatever the runtime put there when it loaded.
	// At<NotifyFn>(runtime, rva->trampoline) = &AlreadySubmitted;
	At<ID3D12Device*>(runtime, rva->device) = device12.get();
	device12.get()->AddRef();
	At<ID3D12CommandQueue*>(runtime, rva->queue) = queue.get();
	queue.get()->AddRef();
	At<int>(runtime, rva->hipDevice) = hipDevice;
	// Which of these writes is the one. Everything below is a plain store at an address from the
	// layout table, so the group that kills it names the field, and the field names the
	// mechanism.
	// Inline stays pinned, and it is the one engine key this backend does not expose.
	//
	// It was unpinned once to try the runtime's asynchronous route, and the attempt is worth
	// recording because its failure was not the engine's. With Inline=0 the completion
	// signal this backend polls -- the sync counter at +0x76c14 against the job counter at
	// +0x76d74 -- never moves: 473 samples, the counter zero in every one, while the
	// runtime's own log showed its worker finishing jobs of 31 to 47 ms throughout. With
	// every wait failing, every frame left through the timeout path below, which returns
	// without closing the command list or writing the output texture, and the screen went
	// black. So the black frame that was blamed on the asynchronous route was this backend
	// never writing a frame.
	//
	// The candidate that looked like a completion counter, +0x76d7c, trailed the job counter
	// by exactly one across 237 samples under a load that should have made it drift -- a
	// frame rate of 34 against jobs of 31 to 47 ms -- so it is a saved copy of the previous
	// job number, not a count of finished ones. Driving the asynchronous route needs a
	// different completion mechanism and a composite that lags a frame, since the engine
	// rewrites its surface in place and a lagged residual has to be paired with the baseline
	// it was measured against. Neither is here, so the pin stays.
	// The engine reads the frame through externally-shared memory, and this is the flag that
	// says so. Without it the runtime has no way into the surfaces this backend hands it, and
	// what comes back is a black frame -- the failure the reference installation never reaches
	// because it sets this before init.
	// One, as the reference sets it. Turned off once to see whether the external-memory interop
	// it enables was what stopped this device executing submitted work: it is not. With this at
	// zero the runtime falls back to "inputs readback, output upload; cpu staging" and every
	// D3D12 submission still produces nothing, so interop is exonerated and the zero-copy path
	// is kept, being both faster and what every working installation runs.
	At<uint8_t>(runtime, rva->interop) = 1;
	// -1 is what the reference passes: the tonemap is the engine's own choice rather than one
	// of the fixed curves, which is what every working installation runs.
	At<int>(runtime, rva->tonemap) = -1;
	// Inline is the mode the runtime ships and the one every working installation runs, but it
	// depends on the driver letting a store from the queue be seen by the GPU-side wait, and
	// the engine says so itself when it cannot: "Inline mode will time out on this driver".
	// Pinned to 1 it is a black frame on such a driver with no way out from the ini, because
	// this write lands after the runtime has read it. Left switchable so the fallback the
	// engine names is reachable:   [DlssNrOnAmd]   Inline=0
	At<uint8_t>(runtime, rva->inlineMode) =
		GetPrivateProfileIntW(L"DlssNrOnAmd", L"Inline", 1,
			(ExeDirectory() / kIniName).c_str()) != 0 ? 1 : 0;
	At<uint8_t>(runtime, rva->enabled) = 1;
	// Deliberately *not* written here: Interop, which belongs to dlssnr_on_amd.ini the way
	// Tonemap does. An earlier version of this function still pinned it to 1 while the
	// comment beside it claimed otherwise, so `Interop=0` in the file did nothing.
	//
	// Inline stays pinned, and it is the one engine key this backend does not expose. The
	// engine in inline mode completes the job on the frame it was given, which is what the
	// rest of this backend assumes when it converts the result out; the asynchronous mode
	// hands back the previous frame instead and the chain has no such path.

	// The engines decides at staging time which guides it will carry, and the staging is
	// built inside the init call below -- so the flags that choose them have to be set
	// before it, not only per pass. This is not a style choice: UseDepth was moved to the
	// per-pass block alone, the staging then read the ini default of 1, and every job since
	// has carried a depth guide this backend has no source for -- fed from a zero-filled
	// texture, and paid for on every frame.
	At<uint8_t>(runtime, rva->useDepth) = settings.amdUseDepth ? 1 : 0;
	At<uint8_t>(runtime, rva->useFsrInputs) = settings.amdUseFsrInputs ? 1 : 0;
	At<uint8_t>(runtime, rva->perPassFlag) = settings.amdTemporal ? 1 : 0;

	// Deliberately *not* written: the tonemap mode at +0x76e20.
	//
	// The reference pins it to -1, "auto tonemap by input format", and this backend copied
	// that. But the runtime reads the same field from its own ini in DllMain, so writing it
	// here overrides whatever the file says -- the same trap the inline and interop flags
	// turned out to be. Left alone, the engine's own default for a missing key is -1, which
	// is what this wrote anyway, so a default install is unaffected and
	// `[DlssNrOnAmd] Tonemap=<n>` in dlssnr_on_amd.ini now has the last word.
	//
	// This matters because the look is reported as a very heavy tone curve -- shadows
	// crushed, highlights blown, the picture reading like a strong cinematic grade -- while
	// the frame mean is unchanged (0.3862 against 0.3846 with the engine at 960x540, which
	// is a contrast curve, not a brightness shift). The tonemap is the field that decides
	// how the network's linear output is presented, so it is the right thing to be able to
	// sweep from a file rather than from a rebuild.

	// Inline mode is what the runtime ships and what every working installation runs.
	// Both flags are also settable from dlssnr_on_amd.ini under [DlssNrOnAmd], but the ini
	// is read in the runtime's DllMain and these writes land after that, so anything set
	// here wins. That was tried the other way round -- leaving them to the ini so Inline=0
	// could be tested -- and the result was a black frame, so Inline is pinned. Do not unpin
	// it without testing one variable at a time.

	// Where the queue stops executing.
	//
	// The crossing test that runs from CreatePipeline passes, and the one at frame forty fails,
	// so the change is somewhere between them. Everything else between them is CreateSized,
	// CreateReconstruction, and this function -- so running the test on both sides of the init
	// call narrows the window to the call itself and to nothing else.

	// How many times this process has brought an engine up.
	//
	// The guard above stops a second init within one Impl, but the runtime is loaded once for
	// the life of the process and never unloaded, while a re-enabled effect gets a new Impl
	// (_ReleaseNgxConsumers clears the backend list) and so a fresh `engineUp`. The comment
	// above says the second call is what crashed; the guard cannot span the case it is about.
	//
	// Counting is all this does. It changes no behaviour, and it is here so that the next log
	// answers whether a crash happened on the first bring-up or on a later one, rather than the
	// question being argued from the code a second time.
	static uint32_t bringUps = 0;
	++bringUps;
	Logger::Get().Info(fmt::format(
		"DLSSNR AMD: engine bring-up {} in this process; the runtime instance at {:x} is about "
		"to have its init called", bringUps, reinterpret_cast<uintptr_t>(runtime)));

	const std::string weights = weightsPath.string();
	if (!CallInit(init,
		reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(runtime) + rva->initCtx),
		&weights)) {
		Logger::Get().Error("DLSSNR AMD: engine initialisation failed");
		return false;
	}


	At<uint8_t>(runtime, rva->initDone) = 1;
	Engine().started = true;
	engineUp = true;
	return true;
}

bool DlssnrAmdBackend::Impl::CreateExposure() noexcept {
	if (exposureTexture) {
		return true;
	}
	// A default-heap, UAV-capable texture, matching the reference's own exposure surface
	// (`createScratch(p->exposureCopy, 1, 1, DXGI_FORMAT_R32_FLOAT)`), because that is the
	// shape of resource the engine is handed and samples. An earlier attempt put this on the
	// upload heap as a shortcut and the filter stopped contributing entirely -- exposure 0
	// normalises the image to black -- so the heap type is not a detail here.
	if (!CreateEngineTexture(device12.get(), 1, 1, DXGI_FORMAT_R32_FLOAT, exposureTexture)) {
		return false;
	}

	// Fill it once. The reference does this with a compute pass; a copy from an upload
	// buffer lands the same value with less machinery, and the value never changes.
	D3D12_HEAP_PROPERTIES uploadHeap{};
	uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
	D3D12_RESOURCE_DESC sourceDesc{};
	sourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	sourceDesc.Width = 256;  // one row at the alignment a 1x1 R32 copy requires
	sourceDesc.Height = 1;
	sourceDesc.DepthOrArraySize = 1;
	sourceDesc.MipLevels = 1;
	sourceDesc.Format = DXGI_FORMAT_UNKNOWN;
	sourceDesc.SampleDesc.Count = 1;
	sourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	winrt::com_ptr<ID3D12Resource> source;
	if (FAILED(device12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE,
		&sourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
		IID_PPV_ARGS(source.put())))) {
		return false;
	}
	void* mapped = nullptr;
	const D3D12_RANGE written{ 0, sizeof(float) };
	if (FAILED(source->Map(0, &written, &mapped)) || !mapped) {
		return false;
	}
	// The value is read from the runtime's own ini under a key the runtime does not know and
	// therefore ignores. Whether the picture looks right at a given exposure is a judgement
	// that needs eyes on the screen, so it should not cost a rebuild per attempt:
	//
	//   [DlssNrOnAmd]
	//   Exposure=0.8
	//
	// Missing, unparseable or non-positive falls back to kExposureValue.
	float exposureValue = kExposureValue;
	{
		const auto iniPath = ExeDirectory() / kIniName;
		wchar_t buffer[64]{};
		GetPrivateProfileStringW(L"DlssNrOnAmd", L"Exposure", L"", buffer,
			static_cast<DWORD>(std::size(buffer)), iniPath.c_str());
		float parsed = 0.0f;
		if (swscanf_s(buffer, L"%f", &parsed) == 1 && std::isfinite(parsed) &&
			parsed > 0.0f) {
			exposureValue = parsed;
		}
	}
	Logger::Get().Info(fmt::format("DLSSNR AMD: exposure {:.3f}", exposureValue));
	*static_cast<float*>(mapped) = exposureValue;
	source->Unmap(0, nullptr);

	if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.get(), nullptr))) {
		return false;
	}
	Barrier(list.get(), exposureTexture.get(), kStateShaderRead,
		D3D12_RESOURCE_STATE_COPY_DEST);
	D3D12_TEXTURE_COPY_LOCATION to{};
	to.pResource = exposureTexture.get();
	to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	to.SubresourceIndex = 0;
	D3D12_TEXTURE_COPY_LOCATION from{};
	from.pResource = source.get();
	from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	from.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
	from.PlacedFootprint.Footprint.Width = 1;
	from.PlacedFootprint.Footprint.Height = 1;
	from.PlacedFootprint.Footprint.Depth = 1;
	from.PlacedFootprint.Footprint.RowPitch = 256;
	list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
	Barrier(list.get(), exposureTexture.get(), D3D12_RESOURCE_STATE_COPY_DEST,
		kStateShaderRead);
	if (FAILED(list->Close())) {
		return false;
	}
	ID3D12CommandList* lists[] = { list.get() };
	SubmitTo(queue.get(), 1, lists);
	const uint64_t signal = ++fenceValue;
	queue->Signal(fence.get(), signal);
	if (fence->GetCompletedValue() < signal) {
		fence->SetEventOnCompletion(signal, fenceEvent.get());
		if (WaitForSingleObject(fenceEvent.get(), 1000) != WAIT_OBJECT_0) {
			return false;
		}
	}
	// Read it back immediately, while the queue is idle and nothing else is mid-recording.
	// This is the one place a known value can be checked without anything else in the way, so
	// it separates "the readback is unreliable where it is called" from "the surface really is
	// empty" -- two very different problems that looked identical in the profile.
	Logger::Get().Info(fmt::format(
		"DLSSNR AMD EXPOSURE CHECK (wrote {:.3f}): reads {:.4f}", exposureValue,
		Measure(exposureTexture.get(), DXGI_FORMAT_R32_FLOAT, kStateShaderRead)));
	return true;
}

bool DlssnrAmdBackend::Impl::WaitForEngine(uint32_t wanted, uint64_t deadlineMs) noexcept {
	// The network runs on the engine's own worker, so a fence on this queue says nothing
	// about whether a result exists. The engine publishes its progress in its sync
	// counter, and that is what the working implementation polls.
	//
	// The loop spins before it sleeps, and the sleep is a real one. `Sleep(0)` gives up the
	// rest of this thread's slice to a ready thread of the same or higher priority on the
	// same processor, which is not a promise that the engine's worker -- a plain work item
	// whose priority this backend does not set -- gets to run. The process timer is already
	// at a millisecond for the engine's sake, so `Sleep(1)` costs about that much and does
	// guarantee the worker is scheduled. Against jobs of twenty to thirty milliseconds that
	// is not a price worth avoiding, and a stall of a third of a second is.
	//
	// The spin is short and entered every time: a completion that lands during it is taken
	// without giving up the processor at all, which is the common case.
	const uint64_t deadline = GetTickCount64() + deadlineMs;
	uint32_t spins = 0;
	while (At<UINT>(runtime, rva->syncCounter) < wanted) {
		if (GetTickCount64() > deadline) {
			return false;
		}
		if (++spins <= 4096) {
			YieldProcessor();
		} else {
			Sleep(1);
		}
	}
	return true;
}

// One job for the engine: its per-pass state, the packet, the submission, and the wait
// for its worker to publish a result.
//
// False means the frame is finished -- the engine refused it, the command list latched an
// error, or the job did not finish inside its budget -- and the caller should pass the
// frame through. A method rather than a block only because the frame's phases are timed
// around it and the job is the one part of Draw that is a single, self-contained act.
bool DlssnrAmdBackend::Impl::SubmitEngineJob() noexcept {
	auto& p = *this;
	phaseAt[2] = Impl::Qpc();
	// ---- 2. the engine's own state, immediately before it records ----
	// Not set-and-forget: the working implementation rewrites the whole block per pass.
	// 0x76e1d is the engine's `Temporal` key as well as the per-pass flag the reference
	// asserts; the control drives it directly.
	At<uint8_t>(p.runtime, p.rva->perPassFlag) = p.settings.amdTemporal ? 1 : 0;

	// History is only meaningful while consecutive frames agree. The engine's own job
	// counter cannot be the trigger, for the reason the reference gives: recreating staging
	// restarts it, so job 1 can follow job 1 and equality says nothing.
	const uint64_t now = GetTickCount64();
	const bool gap = p.lastSubmitTick != 0 && now - p.lastSubmitTick > 250;
	p.lastSubmitTick = now;
	if (p.resetHistory || gap) {
		p.resetHistory = false;
		// The carried residual describes a picture that is no longer here.
		p.temporalValid = false;
		At<uint8_t>(p.runtime, p.rva->wantHistory) = 0;
		At<void*>(p.runtime, p.rva->history) = nullptr;
	}

	// The engine's spin allowance for this frame, scaled with the pixels it is being asked to
	// edit and clamped at the reference's own bounds. Too small and the runtime's wait budget
	// collapses after a timeout -- which is the staircase the engine's own log shows -- and
	// the reference scales it for exactly that reason.
	if (p.netWidth && p.netHeight) {
		const uint64_t allowance = 262144ull + (uint64_t(p.netWidth) * p.netHeight + 1) / 2;
		At<UINT>(p.runtime, p.rva->watchdog) = static_cast<UINT>(allowance < 262144ull ? 262144ull :
			(allowance > 2097152ull ? 2097152ull : allowance));
	}
	At<UINT>(p.runtime, p.rva->depthInverted) = 0;
	At<uint8_t>(p.runtime, p.rva->depthExplicit) = 1;
	// These are the runtime's own defaults, and an attempt to drive them from Magpie's
	// parameters had to be backed out of. Magpie's UI defaults are the *inverse* of the
	// runtime's for two of the three: its localToneStrength defaults to 1 where the runtime
	// starts at 0, its skinStructureStrength defaults to 0 where the runtime starts near 1,
	// and its useAutoMask defaults to false where the runtime starts at 1. Writing the UI
	// values straight through therefore switched off the engine's auto mask and skin
	// structure on every default install, and the visible edit collapsed: the engine log
	// showed the chain healthy (jobs completing, history on) while the probe read
	// `frame in 0.5502, engine out 0.5525` -- the filter running but editing almost nothing.
	//
	// So the parameter set cannot be mapped one-to-one. Magpie's names and defaults were
	// chosen for the NGX residual path, where they mean something else. Wiring them up is
	// still wanted, but it has to be done one field at a time against a real frame, with
	// the defaults reconciled first -- not all at once on the assumption that the names
	// line up.
	// The engine's own controls, from the effect's parameters rather than from constants.
	// Every one of these was a fixed value until now, which is why the controls in Magpie's
	// UI appeared to do nothing: they were being written over here on every frame.
	// NR Style has no engine field to write. The runtime's own UI lists exactly what it
	// exposes -- Enabled, Tone intensity, Structure intensity, Skin structure and Inline --
	// with no style among them, its fifteen configuration keys have none either, and the
	// name the NGX path uses is an extension key of its own (`DLSSNR.Style`). The runtime
	// also states in its own text that it gates off the broad lighting and colour channels a
	// style would act on, so there is nothing here for a style to reach directly.
	//
	// What the three are given instead is the engine's content controls, chosen by what the
	// network's own description says they do rather than by what the names suggest. Its
	// Structure control adds ambient occlusion, contact shadows, reflections and subsurface
	// scattering, and its skin channel routes structure through a semantic character mask;
	// both are wrong for flat colour and hard edges, which is exactly what makes Default
	// unusable on stylised and animated content. So:
	//
	//   Default    the sliders govern, unchanged
	//   Natural    a light touch: less of the added detail, little of the character channel,
	//              no tone channels -- for animation, cel shading and stylised rendering
	//   Cinematic  more of both, with the tone channels on, for photographic content
	//
	// A scale, not an override, so the sliders stay authoritative at Default and their effect
	// stays visible at the other two.
	const float styleDetail = p.settings.style == 1 ? 0.6f
		: p.settings.style == 2 ? 1.25f : 1.0f;
	const float styleSkin = p.settings.style == 1 ? 0.3f
		: p.settings.style == 2 ? 1.2f : 1.0f;
	const UINT styleChannels = p.settings.style == 1 ? 0u
		: p.settings.style == 2 ? std::max(1u, static_cast<UINT>(p.settings.amdToneChannels))
		: static_cast<UINT>(std::clamp(p.settings.amdToneChannels, 0, 2));

	At<float>(p.runtime, p.rva->localTone) =
		std::clamp(p.settings.localToneStrength, 0.0f, 2.0f);
	const float engineStructure =
		std::clamp(p.settings.localStructureStrength * styleDetail, 0.0f, 2.0f);
	const float engineSkin =
		std::clamp(p.settings.skinStructureStrength * styleSkin, 0.0f, 2.0f);
	At<float>(p.runtime, p.rva->localStructure) = engineStructure;
	At<float>(p.runtime, p.rva->skinStructure) = engineSkin;
	At<UINT>(p.runtime, p.rva->toneChannels) = styleChannels;
	// The values the engine actually receives, so a style that is supposed to change them
	// can be seen doing so rather than inferred.
	if (p.settings.style != p.loggedStyle || p.settings.intensity != p.loggedIntensity) {
		p.loggedStyle = p.settings.style;
		p.loggedIntensity = p.settings.intensity;
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD engine values: style={} -> structure={:.3f} skin={:.3f} "
			"toneChannels={} intensity={:.2f}", p.settings.style, engineStructure,
			engineSkin, styleChannels, p.settings.intensity));
	}
	At<UINT>(p.runtime, p.rva->charMask) = p.settings.useAutoMask ? 1u : 0u;
	At<uint8_t>(p.runtime, p.rva->useDepth) = p.settings.amdUseDepth ? 1 : 0;
	At<uint8_t>(p.runtime, p.rva->useFsrInputs) = p.settings.amdUseFsrInputs ? 1 : 0;
	// Deliberately absent: the wait allowance at +0x76c44.
	//
	// The working implementation writes `262144 + pixels/2` into that field every pass, and
	// this backend did the same thing at first. It is a mistake, and the engine's own log
	// says so plainly. That field is the iteration ceiling the inline wait spins against,
	// and the engine maintains it itself -- its log shows the ceiling at 13659064, then
	// 211732559, 191115557, 220213086, 400000000, changing as it measures. Writing the
	// formula fights it for the field, and the formula's value for 1920x1080 is 1298944:
	// about 3.5 ms of spinning at the ~370000 iterations/ms the engine reports, against
	// jobs that take 250-280 ms.
	//
	// The log makes the split unmistakable. Every pass where the engine held the field
	// finished cleanly -- "spin used 404259 iterations for a 63 ms job", no timeout, at a
	// ceiling in the tens or hundreds of millions. Every one of the thirteen timeouts in
	// that run reports "iteration cap after 1298944 iterations, cap 1298944", which is the
	// formula's number, followed by "current input kept" -- the frame handed back
	// unprocessed. So the intermittent drop-out is not the engine being too slow; it is
	// this backend lowering the engine's own ceiling and then losing the race to restore it.
	// The failure feeds itself, because the engine shortens its wait budget after each
	// timeout, which makes the next expiry likelier under exactly the heavy load where the
	// user notices it.
	//
	// Leaving the field alone lets the engine do what it was already doing: it recovers on
	// its own, logging "100 clean jobs; wait budget back to 600 ms".

	// The heap, signature and table the reference leaves bound when it hands the frame to
	// the engine. Slot 0 is the input as a shader resource and slot 1 the same surface as
	// an unordered-access target -- which is right for a filter that reads and writes one
	// texture. The engine opens the resource for HIP access itself; this binding is here
	// so the command list matches the one the engine was written against.
	p.Bind(0, p.net[p.slot].get(), DXGI_FORMAT_R16G16B16A16_FLOAT,
		p.net[p.slot].get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
	{
		ID3D12DescriptorHeap* h = p.heap.get();
		p.list->SetComputeRootSignature(p.root.get());
		p.list->SetDescriptorHeaps(1, &h);
		p.list->SetComputeRootDescriptorTable(0,
			p.heap->GetGPUDescriptorHandleForHeapStart());
	}

	Packet packet{};
	packet.list = p.list.get();
	// One surface, read and written in place. This is the whole contract: the reference
	// hands over its engine-resolution colour texture and then reads the result out of
	// that same texture. There is no second resource, and pointing this field at one --
	// as an earlier version of this backend did -- leaves the engine staring at an empty
	// texture, which is what its own log reports as "encoded mean 0.000".
	packet.colour = p.net[p.slot].get();
	packet.colourState = kPacketState;
	packet.motion = p.motion[p.slot].get();
	packet.motionState = kPacketState;
	packet.depth = p.depth[p.slot].get();
	packet.depthState = kPacketState;
	// An exposure rather than nullptr, so the engine uses it instead of adapting its own.
	// Its adaptation is not stable; the measurements are on kExposureValue.
	packet.exposure = p.exposureTexture.get();
	packet.exposureState = kPacketState;
	// The motion field's pixel scale relative to the network's, as the reference computes
	// it when it resamples motion for the engine. Without a motion guide these stay at one,
	// where they have always been.
	packet.scaleX = p.motionReady ? p.motionScaleX : 1.0f;
	packet.scaleY = p.motionReady ? p.motionScaleY : 1.0f;
	// The three fields 0.3.0 added to the packet.
	//
	// nativePre stays zero: that is the native FFX pre-SR path, which brings its own
	// input/output semantics this backend does not use -- it hands over its own staging and
	// takes the ordinary path, which is what the reference does for the same reason.
	//
	// The render extent is the size the model actually works at. It is not the capture size,
	// and saying so is the point: the engine needs to know what resolution it is being asked
	// to edit, and until now it was told nothing.
	//
	// Jitter is zero on purpose rather than left to whatever the stack held. This backend
	// samples the network's input at the pixel centre every frame, so there is no sub-pixel
	// phase to declare, and passing an uninitialised value as if there were one would move
	// the picture by an amount nothing asked for.
	packet.nativePre = 0;
	packet.renderWidth = p.netWidth;
	packet.renderHeight = p.netHeight;
	packet.jitterX = 0.0f;
	packet.jitterY = 0.0f;

	// No transitions here. The engine is handed shader-readable surfaces, exactly as the
	// reference hands it shader-readable surfaces, and it deals with hazards itself.

	if (!CallRecord(p.record, &packet) || At<uint8_t>(p.runtime, p.rva->statusFlag) != 0) {
		p.failed = true;
		Logger::Get().Error("DLSSNR AMD: the engine refused the frame");
		return false;
	}

	if (FAILED(p.list->Close())) {
		p.failed = true;
		Logger::Get().Error("DLSSNR AMD: the command list latched an error");
		return false;
	}
	ID3D12CommandList* lists[] = { p.list.get() };
	p.SubmitTo(p.queue.get(), 1, lists);
	if (!CallNotify(p.notify, p.queue.get(), 1, lists)) {
		p.failed = true;
		return false;
	}
	phaseAt[3] = Impl::Qpc();

	// The job number this submission produced, remembered against the slot it went into. It
	// is deliberately not waited on here: the runtime returns as soon as the work is queued
	// ("mode async"), and blocking would put the engine's cost back on the render thread,
	// which is the whole thing the asynchronous mode buys. The wait happens where the result
	// is needed, on the slot that holds it.
	p.slotJob[p.slot] = At<UINT>(p.runtime, p.rva->jobCounter);
	p.slotState[p.slot].Record(GetTickCount64());
	// What the runtime actually published, said once a frame for the first few: which job it
	// thinks it was given, how far it says it has got, and whether it still holds the list.
	if (p.framesSeen < 8) {
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD counters frame {}: jobId={} jobDone={} pending={} (slot {} job {})",
			p.framesSeen,
			At<UINT>(p.runtime, p.rva->jobCounter),
			At<UINT>(p.runtime, p.rva->syncCounter),
			reinterpret_cast<uintptr_t>(At<void*>(p.runtime, p.rva->pendingList)),
			p.slot, p.slotJob[p.slot]));
	}
	++p.framesSeen;

	// Signal where the GPU is so the frame this slot holds can be waited on later. The
	// runtime's kernels ride this same queue, so reaching this value means they are done.
	p.slotFence[p.slot] = ++p.serial;
	p.queue->Signal(p.fence.get(), p.slotFence[p.slot]);
	p.slotState[p.slot].Submit(GetTickCount64());
	// Retire whatever has landed. The reference does this at the end of every submission, and
	// it is what keeps a slot from being held for the rest of the session.
	p.RetireSubmission("submitted");

	phaseAt[4] = Impl::Qpc();

	return true;
}

DlssnrAmdBackend::DlssnrAmdBackend() : _impl(std::make_unique<Impl>()) {}

DlssnrAmdBackend::~DlssnrAmdBackend() = default;

bool DlssnrAmdBackend::IsAvailable() noexcept {
	std::error_code ec;
	const auto dir = ExeDirectory();
	return std::filesystem::exists(dir / kRuntimeName, ec) &&
		std::filesystem::exists(dir / kWeightsName, ec);
}

bool DlssnrAmdBackend::Initialize(
	DeviceResources& resources,
	ID3D11Texture2D* input,
	ID3D11Texture2D* output,
	const DLSSNRSettings& settings,
	int antiFlickerMode
) noexcept {
	auto& p = *_impl;
	// Set before anything is allocated: whether the residual path is in use decides which
	// surfaces exist. Modes 2 to 4 are optical-flow routes and there is no optical flow here,
	// so every non-zero request is served as mode 1 -- static accumulation, the route
	// upstream's design document names as needing no motion. Said out loud, because a
	// setting that quietly does something else is worse than one that says so.
	// One is static accumulation; two and above reproject the carried residual with optical
	// flow, which now exists here. The route is kept as asked rather than collapsed to one,
	// and the pass branches on it.
	p.antiFlickerMode = std::clamp(antiFlickerMode, 0, 4);
	if (p.antiFlickerMode >= 2 && !p.wantMotion) {
		Logger::Get().Warn(
			"DLSSNR AMD: anti-flicker uses optical flow but no motion was requested; "
			"serving it as static accumulation");
	}
	// Recorded, not yet acted on. The engine's fields are still written from this
	// backend's own constants because mapping Magpie's parameter names onto them one-to-one
	// does not work -- see the note in Draw. Printing them means the next attempt starts
	// from what Magpie actually passes rather than from what the UI claims its defaults are.
	Logger::Get().Info(fmt::format(
		"DLSSNR AMD parameters: tone={:.3f} structure={:.3f} skin={:.3f} autoMask={} "
		"temporal={} toneChannels={} useDepth={} useHostInputs={} resolution={}% "
		"style={} intensity={:.2f} reconstruct={}",
		settings.localToneStrength, settings.localStructureStrength,
		settings.skinStructureStrength, settings.useAutoMask ? "on" : "off",
		settings.amdTemporal, settings.amdToneChannels, settings.amdUseDepth,
		settings.amdUseFsrInputs,
		settings.enableInputResolutionScaling ? int(settings.inputResolutionPercent) : 100,
		settings.style, settings.intensity, settings.amdReconstruct));
	// The one control that moves the frame rate. The network's cost is close to linear in
	// the pixels it is handed -- measured at roughly 30 ms per megapixel here -- and in
	// inline mode the game waits for it, so the frame rate is its reciprocal. Magpie's
	// parameter surface already carried this knob for the NGX path, where the residual is
	// composited back at full size; the same idea is done explicitly here.
	//
	// This is the only parameter mapped so far. The others are not wired, deliberately:
	// Magpie's defaults for them are the inverse of the runtime's, so copying them across
	// switched the engine's own filters off and the visible edit collapsed. See the note
	// where the engine's state is written.
	// The effect's two controls own this -- the checkbox and the 25-100% slider -- and they are
	// read exactly as the interface leaves them. The file is only a fallback for a run that
	// cannot reach the interface: an ini value is honoured when the checkbox is off, and
	// ignored the moment someone turns scaling on, so the slider always wins once it is in use.
	//   [DlssNrOnAmd]   InputResolutionPercent=40
	const int scaleFromIni = GetPrivateProfileIntW(L"DlssNrOnAmd", L"InputResolutionPercent",
		0, (ExeDirectory() / kIniName).c_str());
	p.modelScale = settings.enableInputResolutionScaling
		? float(std::clamp<uint32_t>(settings.inputResolutionPercent, 25, 100)) / 100.0f
		: (scaleFromIni > 0 ? float(std::clamp(scaleFromIni, 25, 100)) / 100.0f : 1.0f);

	// Whether the frame goes to the engine as linear values or exactly as it arrives.
	//
	// The engine's own residual shader decodes sRGB and re-encodes it around the edit, which
	// only comes out as the identity if what it is given is linear to begin with; and the
	// reference states the same of its own path -- "color composition is adapted to
	// pre-exposed linear input". Magpie's eight-bit SDR frame is sRGB-encoded, so the two
	// only line up if something converts. Read from the runtime's ini under a key it does
	// not know, so the answer can come from a frame rather than from a rebuild:
	//
	//   [DlssNrOnAmd]
	//   SrgbInput=0
	{
		const auto iniPath = ExeDirectory() / kIniName;
		p.srgbInput =
			GetPrivateProfileIntW(L"DlssNrOnAmd", L"SrgbInput", 1, iniPath.c_str()) != 0;
	}
	Logger::Get().Info(fmt::format("DLSSNR AMD: srgb input {}",
		p.srgbInput ? "yes" : "no"));
	{
		const auto iniPath = ExeDirectory() / kIniName;
		p.encodeSteps = static_cast<int>(GetPrivateProfileIntW(L"DlssNrOnAmd",
			L"EncodeSteps", 5, iniPath.c_str()));
		p.jitterEnabled = GetPrivateProfileIntW(L"DlssNrOnAmd",
			L"Jitter", 1, iniPath.c_str()) != 0;
		p.pointSample = GetPrivateProfileIntW(L"DlssNrOnAmd",
			L"JitterShape", 1, iniPath.c_str()) != 0;
	}

	// How much of the network's edit to apply, in thousandths. Read from the same ini; the
	// working implementation attenuates the edit to stop flicker at reduced scale and lists
	// that as a known quality reduction, so the amount is a judgement to be made on a frame.
	{
		const auto iniPath = ExeDirectory() / kIniName;
		const int bound = GetPrivateProfileIntW(L"DlssNrOnAmd", L"EditBound", 500,
			iniPath.c_str());
		p.editBoundMilli = static_cast<uint32_t>(std::clamp(bound, 5, 1000));
	}
	{
		const auto iniPath = ExeDirectory() / kIniName;
		const int shoulder = GetPrivateProfileIntW(L"DlssNrOnAmd", L"HighlightShoulder",
			850, iniPath.c_str());
		p.shoulderMilli = static_cast<uint32_t>(std::clamp(shoulder, 50, 990));
	}
	// The reconstruction route is an effect parameter rather than an ini key, so the app's own
	// control reaches it and there is one source of truth for it. The ini key it was read from
	// during development is gone rather than kept alongside.
	p.reconstruct = settings.amdReconstruct != 0;
	Logger::Get().Info(fmt::format("DLSSNR AMD: edit bound {} / 1000", p.editBoundMilli));
	// The engine module stays loaded across effect rebuilds and keeps its history, so a
	// new backend instance always starts by invalidating it.
	p.resetHistory = true;
	p.device11 = resources.GetD3DDevice();
	p.context11 = resources.GetD3DDC();
	if (!p.device11 || !p.context11) {
		return false;
	}

	D3D11_TEXTURE2D_DESC inputDesc{}, outputDesc{};
	input->GetDesc(&inputDesc);
	output->GetDesc(&outputDesc);
	if (inputDesc.Width != outputDesc.Width || inputDesc.Height != outputDesc.Height) {
		Logger::Get().Error(fmt::format(
			"DLSSNR AMD: input and output differ in size: {}x{} vs {}x{}",
			inputDesc.Width, inputDesc.Height, outputDesc.Width, outputDesc.Height));
		return false;
	}
	// The formats are allowed to differ, and usually do: Magpie hands a native effect
	// B8G8R8A8 and expects R8G8B8A8 back. Only the sizes have to agree, because the
	// engine works at one resolution.
	const bool inputOk = inputDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
		inputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM ||
		inputDesc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT;
	const bool outputOk = outputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM ||
		outputDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
		outputDesc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT;
	if (!inputOk || !outputOk) {
		Logger::Get().Error(fmt::format(
			"DLSSNR AMD: unsupported formats: input={} output={}",
			(uint32_t)inputDesc.Format, (uint32_t)outputDesc.Format));
		return false;
	}

	// The engine's device and queue come from the process, not from this session. The first
	// session creates them; every session after that adopts them, because the runtime is never
	// unloaded and its engine holds pointers into them. See EngineHost.
	EngineHost& host = Engine();
	if (host.device) {
		const LUID engineLuid = host.device->GetAdapterLuid();
		LUID here{};
		bool hereKnown = false;
		{
			winrt::com_ptr<IDXGIDevice> dxgi;
			winrt::com_ptr<IDXGIAdapter> adapter;
			DXGI_ADAPTER_DESC d{};
			if (SUCCEEDED(p.device11->QueryInterface(IID_PPV_ARGS(dxgi.put()))) &&
				SUCCEEDED(dxgi->GetAdapter(adapter.put())) &&
				SUCCEEDED(adapter->GetDesc(&d))) {
				here = d.AdapterLuid;
				hereKnown = true;
			}
		}
		// This session's shared textures are made on its D3D11 device and read by the engine on
		// the device it was initialised on. Different cards means memory the engine cannot
		// reach, and the engine cannot be moved -- so the session does not start.
		if (hereKnown && std::memcmp(&here, &engineLuid, sizeof(LUID)) != 0) {
			Logger::Get().Error(fmt::format(
				"DLSSNR AMD: this session renders on adapter {:08x}:{:08x} but the engine was "
				"initialised on {:08x}:{:08x}; it cannot move between cards, so the effect will "
				"not start until Magpie is restarted",
				static_cast<uint32_t>(here.HighPart), static_cast<uint32_t>(here.LowPart),
				static_cast<uint32_t>(engineLuid.HighPart),
				static_cast<uint32_t>(engineLuid.LowPart)));
			return false;
		}
		// A removed device fails every call that follows, so it is refused in one place rather
		// than reported from inside a frame. The engine cannot be rebuilt on a replacement
		// either -- that is the re-initialisation this whole mechanism exists to avoid -- so a
		// restart of Magpie is the only way back.
		const HRESULT removed = host.device->GetDeviceRemovedReason();
		if (FAILED(removed)) {
			Logger::Get().Error(fmt::format(
				"DLSSNR AMD: the engine's device was removed (0x{:08x}); it cannot be rebuilt on "
				"another one, so the effect will not start until Magpie is restarted",
				static_cast<uint32_t>(removed)));
			return false;
		}
		p.device12 = host.device;
		p.queue = host.queue;
		Logger::Get().Info(
			"DLSSNR AMD: adopting the engine's device and queue from the first session in this "
			"process; the engine is not re-initialised");
	} else {
		p.device12.attach(CreateDeviceOnAdapter(p.device11));
		if (!p.device12) {
			Logger::Get().Error("DLSSNR AMD: no D3D12 device on the render adapter");
			return false;
		}
		D3D12_COMMAND_QUEUE_DESC qd{};
		qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
		if (FAILED(p.device12->CreateCommandQueue(&qd, IID_PPV_ARGS(p.queue.put())))) {
			Logger::Get().Error("DLSSNR AMD: could not create the engine's command queue");
			return false;
		}
		host.device = p.device12;
		host.queue = p.queue;
	}

	if (FAILED(p.device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
			IID_PPV_ARGS(p.allocator.put()))) ||
		FAILED(p.device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
			p.allocator.get(), nullptr, IID_PPV_ARGS(p.list.put()))) ||
		FAILED(p.device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
			IID_PPV_ARGS(p.outAllocator.put()))) ||
		FAILED(p.device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
			p.outAllocator.get(), nullptr, IID_PPV_ARGS(p.outList.put()))) ||
		FAILED(p.device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
			IID_PPV_ARGS(p.measureAllocator.put()))) ||
		FAILED(p.device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
			p.measureAllocator.get(), nullptr, IID_PPV_ARGS(p.measureList.put()))) ||
		FAILED(p.measureList->Close()) ||
		// A list is created open and must be closed once before it can ever be reset.
		FAILED(p.outList->Close()) ||
		FAILED(p.device12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
			IID_PPV_ARGS(p.fence.put())))) {
		Logger::Get().Error("DLSSNR AMD: could not create the D3D12 command objects");
		return false;
	}

	// The queue's own ExecuteCommandLists, taken now -- before the runtime's DLL is loaded.
	//
	// It used to be taken at the top of InitEngine, and that was too late to mean anything: the
	// runtime hooks this entry when its DLL loads, which happens well before InitEngine, so what
	// was saved there was the hook rather than the original. Saving it here, while the vtable
	// still belongs to the driver, is the only way to have an entry that actually executes --
	// and the reason it matters is that the runtime's own entry stops executing anything at all
	// once this backend has written its device, queue and enabled flags into the module.
	//
	// Taken once for the process, like the queue it comes from. A later session cannot re-derive
	// it: by then the runtime is loaded, which is precisely the state it has to be read before.
	if (host.executeOriginal) {
		p.executeOriginal = reinterpret_cast<DlssnrAmdBackend::Impl::ExecuteFn>(
			host.executeOriginal);
	} else {
		p.executeOriginal = reinterpret_cast<DlssnrAmdBackend::Impl::ExecuteFn>(
			(*reinterpret_cast<void***>(p.queue.get()))[10]);
		host.executeOriginal = reinterpret_cast<void*>(p.executeOriginal);
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD: the queue's own ExecuteCommandLists is {:x}, taken before the runtime "
			"loads and kept for the process",
			reinterpret_cast<uintptr_t>(p.executeOriginal)));
	}
	// Record which adapter each side is on, once. Everything shared between them assumes the
	// two are the same piece of hardware.
	{
		winrt::com_ptr<IDXGIDevice> dxgi;
		if (SUCCEEDED(p.device11->QueryInterface(IID_PPV_ARGS(dxgi.put())))) {
			winrt::com_ptr<IDXGIAdapter> adapter;
			if (SUCCEEDED(dxgi->GetAdapter(adapter.put()))) {
				DXGI_ADAPTER_DESC d{};
				if (SUCCEEDED(adapter->GetDesc(&d))) {
					p.luid11 = d.AdapterLuid;
					p.device11Adapter = adapter;
				}
			}
		}
		p.luid12 = p.device12->GetAdapterLuid();
	}
	p.fenceEvent.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
	if (!p.fenceEvent.valid() || FAILED(p.list->Close())) {
		return false;
	}

	// Raise the system timer to 1 ms for this process, and hold it for the backend's life.
	//
	// The runtime waits in milliseconds -- its own decompilation shows Sleep(0) and Sleep(1)
	// in the job and worker paths -- and Windows resolves any millisecond wait to the system
	// timer tick, 15.6 ms by default. That tick is what the engine's own job times are
	// quantized to: 15, 31, 46, 62 and 78 ms, all multiples of it, and identical at 480x270
	// and 960x540 despite four times the pixels. The network is not taking 31 ms to do 0.13
	// megapixels of work; its waits are being billed in 15.6 ms units. Since Windows 10 2004
	// the timer resolution is per process, and the engine is a DLL in this process, so
	// raising it here is what those waits need.
	timeBeginPeriod(1);
	p.timerRaised = true;

	// The motion guide's crossing from D3D11 to this device, the same boundary the NGX path
	// uses. It is created once; each frame either updates it with this frame's guidance or
	// leaves the engine on a zero field.
#ifdef MP_ENABLE_AMD_OPTICAL_FLOW
	p.wantMotion = settings.motionRequest.method != OpticalFlowMethod::None;
#else
	// The provider is compiled out of this build; the guidance service would fail the whole
	// session if asked for it.
	p.wantMotion = false;
#endif
	p.motionRequest = settings.motionRequest;
	p.settings = settings;
	// Measurement overrides for the residual controls, in the file the rest of this backend's
	// knobs live in. A missing key leaves the interface's value alone. They exist because
	// those controls are a no-op at their defaults: a wiring mistake in the constant buffer
	// -- the wrong type, the wrong slot -- looks exactly like correct wiring until something
	// moves off 1.0, and these are what move it without the interface. Thousandths, because
	// the profile API reads integers:
	//   [DlssNrOnAmd]   ResidualMultiplierMilli=1500
	{
		const auto over = [&](const wchar_t* key, float& target) {
			const int milli = GetPrivateProfileIntW(L"DlssNrOnAmd", key, -1,
				(ExeDirectory() / kIniName).c_str());
			if (milli >= 0) target = std::clamp(float(milli) / 1000.0f, 0.0f, 4.0f);
		};
		over(L"ResidualMultiplierMilli", p.settings.residualMultiplier);
		over(L"ResidualSaturationMilli", p.settings.residualSaturation);
		over(L"ResidualLightnessMilli", p.settings.residualLightness);
		over(L"ShadowStructureMilli", p.settings.shadowStructureMultiplier);
		over(L"ReflectionGlowMilli", p.settings.reflectionGlowMultiplier);
	}
	p.guidanceInterop = std::make_unique<FrameGuidanceD3D12Interop>();
	if (!p.guidanceInterop->Initialize(p.device12.get(), p.fence.get())) {
		Logger::Get().Error("DLSSNR AMD: guidance interop failed to initialise");
		return false;
	}

	if (!p.LoadRuntime()) {
		return false;
	}

	// The kernel-substitution drop-in (dlssnr_ours.dll) is retired, and deliberately not
	// loaded even when the file is present.
	//
	// It worked by patching dlssnr_amd_pass1.dll's own HIP import slots and running this
	// project's kernels in place of the runtime's. Every address it patches is a 0.4.x
	// address, so against the 0.5.0 runtime it would rewrite slots that no longer mean what
	// it thinks they mean -- and it does that at load time, before anything can check.
	//
	// It also no longer has anything to buy. It existed to work around the runtime's
	// register-resident kernels being `s_trap` stubs on gfx1100; 0.5.0 compiles them, and
	// measured 82 ms -> 27 ms of network time on this card, which is most of what the
	// substitution was written to recover and it arrives without a single address this
	// backend has to maintain.

	if (!p.CreateHip() || !p.CreatePipeline()) {
		return false;
	}
	if (!p.CreateSized(inputDesc.Width, inputDesc.Height,
			inputDesc.Format, outputDesc.Format)) {
		return false;
	}
	if (!p.CreateReconstruction(resources, output)) {
		return false;
	}
	if (!p.InitEngine(ExeDirectory() / kWeightsName)) {
		return false;
	}

	// The crossing, again, with the engine initialised and before a single frame has been drawn.
	//
	// The first crossing test runs before the engine initialises and the second at frame forty,
	// so they differ in two things at once: whether the engine is live, and whether Magpie has
	// been capturing and scaling. This one holds the second still. If the crossing is already
	// dead here, the engine's initialisation is what kills it; if it is alive here and dead at
	// frame forty, the engine is exonerated and something the frames do is responsible.

	p.ready = true;
	// The applied scale is printed alongside the requested one, because the floor can raise
	// it and a setting that silently does nothing is worse than one that says so.
	const float appliedScale = float(p.netWidth) / float(inputDesc.Width);
	Logger::Get().Info(fmt::format(
		"DLSSNR AMD: engine up on HIP device {}; {}x{} {} -> {}, network at {}x{} "
		"(requested {:.2f}, applied {:.2f}), anti-flicker={}, motion={}",
		p.hipDevice, inputDesc.Width, inputDesc.Height,
		(uint32_t)inputDesc.Format, (uint32_t)outputDesc.Format,
		p.netWidth, p.netHeight, p.modelScale, appliedScale, p.antiFlickerMode,
		p.wantMotion ? "optical-flow" : "none"));

	// What the chain hands this effect, which decides how the FSR3 route has to be built.
	//
	// The upscaler backends in this project take a render-resolution frame and reconstruct the
	// display one: FSR3Upscaler dispatches with renderSize set to its input and writes its
	// output, and it creates its own flat depth, so a path with no depth source is already
	// inside its contract. Running the network at a reduced size and reconstructing afterwards
	// would replace the resolve, which composites an upsampled residual onto the captured
	// frame instead, and it is the reconstruction quality that is wanted.
	//
	// Whether that is a chaining job or an embedding one turns on one fact. ResizeTextures
	// hands each effect's output to the next, sized from that effect's descriptor, and this
	// effect declares its output at the input's size. If the chain ever hands over an output
	// smaller than the input, the reconstruction can simply be the next effect in the chain,
	// and the work is a declaration plus a mode. If it never does, the upscaler has to be
	// built inside this backend. Logged rather than assumed: the two differ by an order of
	// magnitude, and one run settles it.
	{
		std::error_code ec;
		const bool upscalerPresent = std::filesystem::exists(
			ExeDirectory() / L"amd_fidelityfx_upscaler_dx12.dll", ec);
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD chain sizes: input {}x{} -> output {}x{} ({}); network {}x{}, "
			"resolve {}; FSR3 upscaler runtime {}",
			inputDesc.Width, inputDesc.Height, outputDesc.Width, outputDesc.Height,
			(inputDesc.Width == outputDesc.Width && inputDesc.Height == outputDesc.Height)
				? "equal, the chain asks for no resize here" : "different",
			p.netWidth, p.netHeight, p.scaled ? "in use" : "not in use",
			upscalerPresent ? "present" : "missing"));
	}
	return true;
}

bool DlssnrAmdBackend::Resize(
	DeviceResources& resources, ID3D11Texture2D* input, ID3D11Texture2D* output
) noexcept {
	(void)resources;
	auto& p = *_impl;
	if (!p.ready) {
		return false;
	}
	D3D11_TEXTURE2D_DESC inDesc{}, outDesc{};
	input->GetDesc(&inDesc);
	output->GetDesc(&outDesc);
	if (inDesc.Width == p.width && inDesc.Height == p.height &&
		inDesc.Format == p.inputFormat && outDesc.Format == p.outputFormat) {
		return true;
	}

	// Both devices have to be idle before the surfaces they share are replaced.
	if (p.fence) {
		const uint64_t signal = ++p.fenceValue;
		p.queue->Signal(p.fence.get(), signal);
		if (p.fence->GetCompletedValue() < signal) {
			p.fence->SetEventOnCompletion(signal, p.fenceEvent.get());
			WaitForSingleObject(p.fenceEvent.get(), 1000);
		}
	}
	const bool sized =
		p.CreateSized(inDesc.Width, inDesc.Height, inDesc.Format, outDesc.Format);
	if (!sized) {
		return false;
	}
	// Rebuilt against the new sizes; CreateSized has already released the old surfaces, so
	// this is the only place the upscaler can be handed a matching pair.
	return p.CreateReconstruction(resources, output);
}

float DlssnrAmdBackend::Impl::MeanOfOutput(ID3D11Texture2D* tex, float* detail) noexcept {
	D3D11_TEXTURE2D_DESC desc{};
	tex->GetDesc(&desc);
	if (desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM &&
		desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
		// Only the eight-bit display formats are read here; a half-float one would need its own
		// decode and this check exists for the eight-bit path.
		return -1.0f;
	}

	D3D11_TEXTURE2D_DESC stagingDesc = desc;
	stagingDesc.Usage = D3D11_USAGE_STAGING;
	stagingDesc.BindFlags = 0;
	stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	stagingDesc.MiscFlags = 0;
	winrt::com_ptr<ID3D11Texture2D> staging;
	if (FAILED(device11->CreateTexture2D(&stagingDesc, nullptr, staging.put()))) {
		return -1.0f;
	}
	context11->CopyResource(staging.get(), tex);

	D3D11_MAPPED_SUBRESOURCE mapped{};
	if (FAILED(context11->Map(staging.get(), 0, D3D11_MAP_READ, 0, &mapped))) {
		return -1.0f;
	}
	// A coarse grid rather than every pixel: this answers whether the picture is roughly where
	// it should be, and a stride keeps a one-off read cheap.
	double sum = 0.0;
	uint64_t count = 0;
	const uint8_t* base = static_cast<const uint8_t*>(mapped.pData);
	const bool bgra = desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM;
	for (uint32_t y = 0; y < desc.Height; y += 16) {
		const uint8_t* row = base + size_t(y) * mapped.RowPitch;
		for (uint32_t x = 0; x < desc.Width; x += 16) {
			const uint8_t* px = row + size_t(x) * 4;
			const double r = bgra ? px[2] : px[0];
			const double g = px[1];
			const double b = bgra ? px[0] : px[2];
			sum += (r + g + b) / (3.0 * 255.0);
			++count;
		}
	}
	// A tile read pixel by pixel, which is what a sharpness figure needs: only neighbouring
	// samples carry the high frequencies that separate a resolved picture from a soft one. The
	// coarse grid above would average them away, which is the one thing this number must not do.
	if (detail) {
		const uint32_t gw = desc.Width < 256 ? desc.Width : 256;
		const uint32_t gh = desc.Height < 256 ? desc.Height : 256;
		const uint32_t x0 = (desc.Width - gw) / 2;
		const uint32_t y0 = (desc.Height - gh) / 2;
		double gradSum = 0.0;
		uint64_t gradCount = 0;
		for (uint32_t y = 0; y < gh; ++y) {
			const uint8_t* row = base + size_t(y0 + y) * mapped.RowPitch;
			for (uint32_t x = 0; x + 1 < gw; ++x) {
				const uint8_t* a = row + size_t(x0 + x) * 4;
				const uint8_t* b = a + 4;
				gradSum += (std::abs(int(a[0]) - int(b[0])) + std::abs(int(a[1]) - int(b[1]))
					+ std::abs(int(a[2]) - int(b[2]))) / (3.0 * 255.0);
				++gradCount;
			}
		}
		*detail = gradCount ? float(gradSum / double(gradCount)) : -1.0f;
	}
	context11->Unmap(staging.get(), 0);
	return count ? float(sum / double(count)) : -1.0f;
}

float DlssnrAmdBackend::Impl::Measure(
	ID3D12Resource* res, DXGI_FORMAT format, D3D12_RESOURCE_STATES before,
	uint64_t* nonFinite, float* meanAbsDelta
) noexcept {
	// Only ever called on the first frame. Reading a texture back means a copy, a
	// submit and a wait, which is far too much per frame -- but once, it is the
	// difference between knowing which stage emitted an empty picture and guessing.
	const uint32_t bpp = format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 4;
	// The resource's own extent, not the capture's. This used to read `width`/`height` --
	// the capture size -- which is correct only while nothing is scaled. Once the network
	// runs at a reduced size the copy below asked for a region larger than the source, which
	// is an invalid copy that leaves the readback buffer partly uninitialised, and half-float
	// interpretation of uninitialised bytes is NaN. Every "engine out" figure measured at a
	// reduced scale before this fix was therefore meaningless, including the reading that
	// made the engine's output look a quarter of its input.
	const D3D12_RESOURCE_DESC measured = res->GetDesc();
	// Rounded up to the 256-byte alignment a placed footprint requires; without that an odd
	// width would make every measurement at that size an invalid copy.
	const uint64_t rowPitch = (uint64_t(measured.Width) * bpp + 255) / 256 * 256;
	const uint64_t total = rowPitch * measured.Height;
	const uint32_t mw = uint32_t(measured.Width), mh = measured.Height;

	// Fresh every call, deliberately.
	//
	// It used to be reused whenever the new total fitted inside the old. A copy that then did
	// not land left the previous texture's contents in place, and the figure that came back was
	// the last surface's value rather than this one's -- two different surfaces of the same size
	// read identically whatever they held, and a chain that produced nothing became
	// indistinguishable from one that worked. One allocation of a few tens of megabytes, once a
	// frame, is not worth that ambiguity.
	{
		probe = nullptr;
		D3D12_HEAP_PROPERTIES hp{};
		hp.Type = D3D12_HEAP_TYPE_READBACK;
		D3D12_RESOURCE_DESC rd{};
		rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		rd.Width = total;
		rd.Height = 1;
		rd.DepthOrArraySize = 1;
		rd.MipLevels = 1;
		rd.Format = DXGI_FORMAT_UNKNOWN;
		rd.SampleDesc.Count = 1;
		rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		if (FAILED(device12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
			D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(probe.put())))) {
			return -1.0f;
		}
		probeBytes = total;
	}

	if (!measureList ||
		FAILED(measureAllocator->Reset()) ||
		FAILED(measureList->Reset(measureAllocator.get(), nullptr))) {
		return -1.0f;
	}
	auto* const probeList = measureList.get();
	Barrier(probeList, res, before, D3D12_RESOURCE_STATE_COPY_SOURCE);

	D3D12_TEXTURE_COPY_LOCATION src{};
	src.pResource = res;
	src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	src.SubresourceIndex = 0;

	D3D12_TEXTURE_COPY_LOCATION dst{};
	dst.pResource = probe.get();
	dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	dst.PlacedFootprint.Footprint.Format = format;
	dst.PlacedFootprint.Footprint.Width = mw;
	dst.PlacedFootprint.Footprint.Height = mh;
	dst.PlacedFootprint.Footprint.Depth = 1;
	dst.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;
	probeList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

	Barrier(probeList, res, D3D12_RESOURCE_STATE_COPY_SOURCE, before);
	if (FAILED(probeList->Close())) {
		return -1.0f;
	}
	// The readback goes out on this backend's own queue, not the engine's: see the note on
	// probeQueue. The engine's queue is ordered before it explicitly, so a surface the frame
	// wrote is still read after the write regardless of which queue the copy runs on.
	if (!probeQueue) {
		D3D12_COMMAND_QUEUE_DESC qd{};
		qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
		if (FAILED(device12->CreateCommandQueue(&qd, IID_PPV_ARGS(probeQueue.put()))) ||
			FAILED(device12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
				IID_PPV_ARGS(probeFence.put())))) {
			return -1.0f;
		}
	}
	probeQueue->Wait(fence.get(), fenceValue);
	ID3D12CommandList* lists[] = { probeList };
	SubmitTo(probeQueue.get(), 1, lists);
	const uint64_t sig = ++probeFenceValue;
	probeQueue->Signal(probeFence.get(), sig);
	if (probeFence->GetCompletedValue() < sig) {
		probeFence->SetEventOnCompletion(sig, fenceEvent.get());
		if (WaitForSingleObject(fenceEvent.get(), 3000) != WAIT_OBJECT_0) {
			return -1.0f;
		}
	}

	void* mapped = nullptr;
	const D3D12_RANGE range{ 0, SIZE_T(total) };
	if (FAILED(probe->Map(0, &range, &mapped))) {
		return -1.0f;
	}
	double sum = 0;
	uint64_t count = 0;
	// Non-finite samples are counted, not averaged. Letting one through turns the whole mean
	// into NaN, which says only "something somewhere was not a number" and hides how much of
	// the surface that was.
	uint64_t bad = 0;
	// When asked, the samples are also compared against the previous call's, which is how
	// the flicker is measured: against a still scene, any difference between one frame and
	// the next is the effect changing its mind.
	if (meanAbsDelta) {
		sample.clear();
	}
	if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
		const uint16_t* p16 = static_cast<const uint16_t*>(mapped);
		for (uint64_t i = 0; i < total / 2; ++i) {
			const float v = HalfToFloat(p16[i]);
			if (std::isfinite(v)) { sum += v; ++count; } else { ++bad; }
			if (meanAbsDelta && std::isfinite(v)) sample.push_back(v);
		}
	} else {
		const uint8_t* p8 = static_cast<const uint8_t*>(mapped);
		for (uint64_t i = 0; i < total; ++i) {
			sum += p8[i] / 255.0; ++count;
			if (meanAbsDelta) sample.push_back(p8[i] / 255.0f);
		}
	}
	if (meanAbsDelta) {
		double diff = 0;
		uint64_t diffCount = 0;
		if (previousSample.size() == sample.size()) {
			for (size_t i = 0; i < sample.size(); ++i) {
				diff += std::abs(double(sample[i]) - double(previousSample[i]));
				++diffCount;
			}
		}
		*meanAbsDelta = diffCount ? float(diff / double(diffCount)) : -1.0f;
		previousSample = sample;
	}
	probe->Unmap(0, nullptr);
	if (nonFinite) {
		*nonFinite = bad;
	}
	return count ? float(sum / double(count)) : -1.0f;
}

FrameGuidanceRequirements DlssnrAmdBackend::GetFrameGuidanceRequirements() const noexcept {
	if (!_impl || _impl->failed) {
		return {};
	}
#ifdef MP_ENABLE_AMD_OPTICAL_FLOW
	// The zero view too, the way upstream asks for it: the temporal pass wants a valid zero
	// field to fall back to when the real one is not produced for a frame.
	FrameGuidanceRequirements result{ .zero = true };
	result.Add(_impl->motionRequest);
	return result;
#else
	// This build compiles the AMD optical flow provider as a stub, so asking for motion
	// makes the renderer's guidance service fail its initialization and abort the whole
	// scaling session. Asking for nothing keeps the session running exactly as it did
	// before any of this existed.
	return {};
#endif
}

bool DlssnrAmdBackend::Draw(const NativeEffectDrawContext& context) noexcept {
	auto& p = *_impl;
	if (!p.ready || p.failed || !context.input || !context.output) {
		return false;
	}

	constexpr D3D12_RESOURCE_STATES kCommon = D3D12_RESOURCE_STATE_COMMON;

	// Move to the other slot for this frame, before anything is recorded. Doing it here
	// rather than at the end means no exit path can skip it -- there are several returns
	// below, and a missed flip would leave two consecutive frames writing the same surface
	// while the engine reads it.
	// Advance to the next slot for this frame, before anything is recorded. Doing it here
	// rather than at the end means no exit path can skip it -- there are several returns
	// below, and a missed advance would leave two consecutive frames writing the same surface
	// while the engine reads it. A slot that is recorded but not yet submitted is skipped
	// rather than taken, so nothing in flight is disturbed.
	for (uint32_t step = 0; step < kSlots; ++step) {
		p.slot = (p.slot + 1u) % kSlots;
		if (!p.slotState[p.slot].BlocksRecord()) break;
	}

	// The sub-pixel phase for this frame, before anything is recorded, so the downsample and
	// the upscaler are told the same one. Zero everywhere but the reconstruction route: the
	// edit route takes the whole-footprint average it has always taken, and a phase would
	// only make its input wander for no one's benefit.
	if (p.reconstruct && p.jitterEnabled) {
		p.frameJitterX = kPhaseX[p.jitterPhase] - 0.5f;
		p.frameJitterY = kPhaseY[p.jitterPhase] - 0.5f;
		p.jitterPhase = (p.jitterPhase + 1) & 7u;
	} else {
		p.frameJitterX = 0.0f;
		p.frameJitterY = 0.0f;
	}

	// The frame's stages, on the performance counter. See phaseAccum in Impl for why this
	// cannot be read off the engine's own job figures.
	auto& phaseAt = p.phaseAt;
	phaseAt[0] = Impl::Qpc();

	// The interval since the last one of these, which is the frame time as the renderer and
	// the game together produced it -- not the part of it spent in here. Measured at the top
	// of a frame, it is therefore the frame that has just finished.
	float frameIntervalMs = 0.0f;
	if (p.lastDrawQpc != 0) {
		frameIntervalMs = static_cast<float>(
			double(phaseAt[0] - p.lastDrawQpc) * Impl::MsPerTick());
		if (p.frameIntervalCount < 128) {
			p.frameIntervalsMs[p.frameIntervalCount++] = frameIntervalMs;
		}
	}
	p.lastDrawQpc = phaseAt[0];

	// One line per slow frame, carrying the same phases the window log averages.
	//
	// The window average says a hitch happened and cannot say what it was made of. A window
	// whose worst frame was 312 ms reported an engine phase 4.5 ms above normal, which is
	// 540 ms of extra time across 120 frames -- more than one hitch, and nothing in the
	// window says whether that was one stall or several. Telling a single 312 ms stall from
	// a scatter of smaller ones is the difference between a capture that stopped delivering
	// frames and a chain that got slower, so the slow frame is reported on its own, with the
	// phases belonging to it.
	constexpr float kSlowFrameMs = 80.0f;
	if (frameIntervalMs > kSlowFrameMs && p.slowFramesLogged < 200) {
		++p.slowFramesLogged;
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD slow frame: {:.1f} ms, frameId {} job {} | convert {:.2f} + motion "
			"{:.2f} + record {:.2f} + engine {:.2f} + composite {:.2f} + copy {:.2f} = "
			"{:.2f} ms, {:.2f} elsewhere",
			frameIntervalMs, p.prevFrameId, p.prevFrameJob,
			p.prevFramePhasesMs[0], p.prevFramePhasesMs[1], p.prevFramePhasesMs[2],
			p.prevFramePhasesMs[3], p.prevFramePhasesMs[4], p.prevFramePhasesMs[5],
			p.prevFrameTotalMs, frameIntervalMs - p.prevFrameTotalMs));
	}

	// A texture shared between two devices sits in COMMON on the D3D12 side and has to
	// go back there before the other device may touch it again, so every crossing below
	// is COMMON -> in use -> COMMON.
	// Magpie's frame into the shared surface, through the shader rather than a copy.
	//
	// The SRV is rebuilt only when the frame the renderer hands over is a different texture,
	// which is the usual case but not a guaranteed one, and it is cached because building a
	// view per frame is pure overhead at this point in the frame.
	// Filled by copy rather than by the compute shader that used to do it.
	//
	// The shader was a straight texel-for-texel copy -- dst[id.xy] = src.Load(id.xy) -- and the
	// shared surface is created with the input's own format, so this says the same thing. What
	// differs is whether the far device ever sees it. The crossing test fills a surface by a
	// staging copy and D3D12 reads the value back exactly; the same surface written through a
	// UAV reads as zero on D3D12 however long the fence is held. A UAV write into a surface
	// shared with a second device has to be made visible by the driver, and on this one it is
	// not -- which is why the network's input arrived empty and every stage downstream made a
	// faithful black picture out of it.
	p.context11->CopyResource(p.sharedIn11.get(), context.input);
	// The input crossing, in the only order that works.
	//
	// This is the direction the backend actually depends on -- Magpie's frame is written into a
	// shared surface on D3D11 and read out of it on D3D12 -- and it is the one that was missing.
	// The flush is what submits the fill; the signal behind it is what the other device waits
	// on. Without that wait the copy below reads a surface nothing has written yet: at startup
	// that is nothing at all, and every stage downstream faithfully makes black out of it.
	//
	// Being on the same machine does not order the two queues, and a flush is a submission
	// rather than a completion. Nothing here is a substitute for the fence.
	const uint64_t inputSignal = ++p.inFenceValue;
	if (FAILED(p.context11x->Signal(p.inFence11.get(), inputSignal))) {
		Logger::Get().Error("DLSSNR AMD: the input crossing could not be signalled");
		p.failed = true;
		return false;
	}
	p.context11->Flush();

	if (FAILED(p.allocator->Reset()) || FAILED(p.list->Reset(p.allocator.get(), nullptr))) {
		return false;
	}

	// The far side of that crossing. Recorded into the list rather than waited on here, so the
	// ordering costs the GPU a dependency instead of costing the frame its latency.
	p.queue->Wait(p.inFence12.get(), inputSignal);

	// ---- 1. Magpie's frame into a full-resolution RGBA16F surface ----
	// The engine's format is RGBA16F and Magpie's usually is not, so this is where the two
	// are reconciled. The result is kept at full size whether or not the network will see
	// it: at a reduced scale the resolve needs the untouched full-resolution frame to put
	// its detail back.
	if (p.inputIsFp16) {
		Barrier(p.list.get(), p.sharedIn12.get(), kCommon, D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(p.list.get(), p.full[p.slot].get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_DEST);
		p.list->CopyResource(p.full[p.slot].get(), p.sharedIn12.get());
		Barrier(p.list.get(), p.sharedIn12.get(), D3D12_RESOURCE_STATE_COPY_SOURCE, kCommon);
		Barrier(p.list.get(), p.full[p.slot].get(), D3D12_RESOURCE_STATE_COPY_DEST,
			kStateShaderRead);
	} else {
		Barrier(p.list.get(), p.sharedIn12.get(), kCommon, kStateShaderRead);
		Barrier(p.list.get(), p.full[p.slot].get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		p.Bind(2, p.sharedIn12.get(), p.inputFormat, p.full[p.slot].get(),
			DXGI_FORMAT_R16G16B16A16_FLOAT);
		p.Dispatch(p.srgbInput ? p.convertInSrgb.get() : p.convertIn.get(), 2);
		Barrier(p.list.get(), p.sharedIn12.get(), kStateShaderRead, kCommon);
		Barrier(p.list.get(), p.full[p.slot].get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kStateShaderRead);
	}

	// ---- 1b. the frame the network actually sees ----
	// At a reduced scale this is an area average rather than a bilinear sample, because a
	// point sample loses narrow bright features once the model runs well below the input --
	// the reference makes the same choice for the same reason. The copy into `baseline` has
	// to happen before the engine runs, since the engine edits its surface in place and the
	// resolve needs the difference.
	if (p.downscaling) {
		Barrier(p.list.get(), p.net[p.slot].get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		p.Bind(4, p.full[p.slot].get(), DXGI_FORMAT_R16G16B16A16_FLOAT, p.net[p.slot].get(),
			DXGI_FORMAT_R16G16B16A16_FLOAT);
		const UINT dims[7]{ p.netWidth, p.netHeight, p.width, p.height,
			p.reconstruct && p.pointSample ? 1u : 0u,
			BitsOfFloat(p.frameJitterX), BitsOfFloat(p.frameJitterY) };
		p.DispatchSized(p.downsample.get(), 4, p.netWidth, p.netHeight, 0, dims, 7);
		Barrier(p.list.get(), p.net[p.slot].get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kStateShaderRead);
	} else if (p.scaled) {
		Barrier(p.list.get(), p.full[p.slot].get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(p.list.get(), p.net[p.slot].get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_DEST);
		p.list->CopyResource(p.net[p.slot].get(), p.full[p.slot].get());
		Barrier(p.list.get(), p.net[p.slot].get(), D3D12_RESOURCE_STATE_COPY_DEST,
			kStateShaderRead);
		Barrier(p.list.get(), p.full[p.slot].get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
			kStateShaderRead);
	}
	if (p.scaled) {
		Barrier(p.list.get(), p.net[p.slot].get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(p.list.get(), p.baseline[p.slot].get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_DEST);
		p.list->CopyResource(p.baseline[p.slot].get(), p.net[p.slot].get());
		Barrier(p.list.get(), p.baseline[p.slot].get(), D3D12_RESOURCE_STATE_COPY_DEST,
			kStateShaderRead);
		Barrier(p.list.get(), p.net[p.slot].get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
			kStateShaderRead);
	}

	phaseAt[1] = Impl::Qpc();
	// ---- 1c. the motion guide ----
	// Magpie's frame guidance carries motion vectors in source pixels from its optical-flow
	// provider, and the engine's temporal machinery wants them: with zero motion its history
	// alignment is wrong wherever anything moves. The guidance is resampled to the network's
	// extent with the vectors unchanged, and the packet below converts their pixel scale --
	// the reference's own convention for this exact hand-off.
	p.motionReady = false;
	if (p.wantMotion) {
		const FrameGuidanceExtent extent{ p.width, p.height };
		const FrameGuidanceView guidance = SelectFrameGuidanceChannels(
			context.frameGuidance, context.zeroFrameGuidance, context.frameId,
			extent, true);
		winrt::com_ptr<ID3D11DeviceContext4> context4;
		p.context11->QueryInterface(IID_PPV_ARGS(context4.put()));
		if (context4 && p.guidanceInterop->WaitForProducer(context4.get(), guidance) &&
			p.guidanceInterop->Update(guidance, context.frameId, extent)) {
			const uint32_t sourceW = guidance.motion.metadata.sourceExtent.width;
			const uint32_t sourceH = guidance.motion.metadata.sourceExtent.height;
			p.guidanceInterop->Transition(p.list.get(), D3D12_RESOURCE_STATE_COMMON,
				kStateShaderRead);
			Barrier(p.list.get(), p.motion[p.slot].get(), kStateShaderRead,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			p.Bind(20, p.guidanceInterop->Motion(), DXGI_FORMAT_R16G16_FLOAT,
				p.motion[p.slot].get(), DXGI_FORMAT_R16G16_FLOAT);
			const UINT dims[4]{ p.netWidth, p.netHeight, sourceW, sourceH };
			p.DispatchSized(p.motionResample.get(), 20, p.netWidth, p.netHeight, 0, dims);
			Barrier(p.list.get(), p.motion[p.slot].get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
				kStateShaderRead);
			p.guidanceInterop->Transition(p.list.get(), kStateShaderRead,
				D3D12_RESOURCE_STATE_COMMON);
			p.motionScaleX = float(p.netWidth) / float(std::max(sourceW, 1u));
			p.motionScaleY = float(p.netHeight) / float(std::max(sourceH, 1u));
			p.motionReady = true;
		}
	}

	// Diagnostic A/B: with this file present the engine is skipped entirely and the
	// frame goes straight through both conversions. If the picture is then correct, the
	// conversion chain is sound and the empty output comes from the engine; if it is
	// still black, the fault is here and the engine was never involved.
	static const bool kBypassEngine = [] {
		std::error_code ec;
		return std::filesystem::exists(ExeDirectory() / L"dlssnr_amd_bypass.txt", ec);
	}();
	if (kBypassEngine) {
		// The frame goes through the same output conversion the real path uses, reading
		// the engine's input texture instead of its output. An earlier version of this
		// copied the engine's surface straight into sharedOut12, which is FP16 into RGBA8
		// -- an invalid copy that leaves the destination untouched, so the screen showed
		// uninitialised memory. That is the pink noise, and it came from the diagnostic,
		// not the chain.
		Barrier(p.list.get(), p.sharedOut12.get(), kCommon,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		// Bypassed: the full-resolution converted frame, not the network's surface, which at
		// a reduced scale is the wrong size for this and would be a format-sized mismatch.
		p.Bind(6, p.full[p.slot].get(), DXGI_FORMAT_R16G16B16A16_FLOAT,
			p.sharedOut12.get(), p.outputFormat);
		p.list->SetComputeRoot32BitConstants(1, 1, &p.shoulderMilli, 0);
		p.Dispatch(p.srgbInput ? p.convertOutSrgb.get() : p.convertOut.get(), 6);
		Barrier(p.list.get(), p.sharedOut12.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kCommon);
		p.list->Close();
		ID3D12CommandList* bl[] = { p.list.get() };
		p.SubmitTo(p.queue.get(), 1, bl);
		const uint64_t bs = ++p.fenceValue;
		p.queue->Signal(p.fence.get(), bs);
		if (p.fence->GetCompletedValue() < bs) {
			p.fence->SetEventOnCompletion(bs, p.fenceEvent.get());
			WaitForSingleObject(p.fenceEvent.get(), 1000);
		}
		p.context11->CopyResource(context.output, p.sharedOut11.get());

		// The bypass returns before the main probe, so it needs its own: this is what says
		// whether the output stage works when the engine is taken out of the picture.
		if (!p.probed && p.framesSeen >= 40) {
			p.probed = true;
			Logger::Get().Info(fmt::format(
				"DLSSNR AMD PROBE (bypass, engine skipped): magpie-input {:.4f} | shared-output {:.4f}",
				p.MeanOfOutput(p.sharedIn11.get()), p.MeanOfOutput(p.sharedOut11.get())));
		}

		if (!p.probed && ++p.framesSeen >= 10) {
			p.probed = true;
			const float inMean = p.Measure(p.sharedIn12.get(), p.inputFormat,
				D3D12_RESOURCE_STATE_COMMON);
			const float netMean = p.Measure(p.full[p.slot].get(),
				DXGI_FORMAT_R16G16B16A16_FLOAT, kStateShaderRead);
			const float outMean = p.Measure(p.sharedOut12.get(), p.outputFormat,
				D3D12_RESOURCE_STATE_COMMON);
			Logger::Get().Info(fmt::format(
				"DLSSNR AMD probe (bypassed): frame in {:.4f}, converted {:.4f}, "
				"converted out {:.4f}", inMean, netMean, outMean));
		}
		return true;
	}

	if (!p.SubmitEngineJob()) {
		return true;
	}

	if (p.reconstruction) {
		// The reconstruction writes the output itself -- it is handed the engine's frame and
		// the output texture, and does its own copying and fencing between them. Everything
		// below composes the edit back here instead, so none of it should run.
		//
		// A failure is latched rather than retried: the output texture would keep whatever it
		// held, which is a frozen picture, and a route that cannot draw should say so through
		// the same rejection any other broken effect gets.
		if (!p.RunReconstruction(context)) {
			if (++p.reconstructionFailures <= 3) {
				Logger::Get().Warn(fmt::format(
					"DLSSNR AMD: the reconstruction did not draw ({} so far)",
					p.reconstructionFailures));
			}
			p.failed = true;
		}
		return true;
	}

	// ---- 3. back out to Magpie's frame ----
	const uint64_t signal = ++p.fenceValue;
	p.queue->Signal(p.fence.get(), signal);
	if (p.fence->GetCompletedValue() < signal) {
		p.fence->SetEventOnCompletion(signal, p.fenceEvent.get());
		if (WaitForSingleObject(p.fenceEvent.get(), 1000) != WAIT_OBJECT_0) {
			p.failed = true;
			return true;
		}
	}

	// The engine leaves its surface readable, which is the state named below.
	//
	// At a reduced scale a resolve sits here: only the difference the engine made is
	// upscaled, bounded, and weighted by how well the full-resolution original agrees with
	// the low-resolution footprint. The picture is therefore the untouched full-resolution
	// frame plus that edit -- which is what keeps a smaller model from costing detail. At
	// 100% there is nothing to resolve and the engine's own surface is the picture.
	// Which slot the picture comes from.
	//
	// The engine was handed this frame's surface and returned without waiting for it, so the
	// frame it has actually finished is the previous one -- "residual from an earlier frame"
	// is what the runtime calls it. Reading the slot just submitted would take a surface the
	// engine is still writing, which is black.
	//
	// The first frames are the exception: there is no earlier frame yet, so the picture waits
	// for the one being submitted. That is also the only place a wait happens now, and it
	// waits for a job number rather than an empty queue.
	// The newest generation the engine has actually finished, across every slot.
	//
	// This is what the slots are for. The engine takes about three frames here, so the frame just
	// submitted is never ready, and waiting on it ties this frame to the engine's rate -- that is
	// the stutter, not the throughput. Take the newest that has retired instead: the edit is then
	// a frame or two old, which is what asynchronous mode means and what the reference does, and
	// the frame time belongs to the renderer again.
	//
	// A slot counts as retired once the fence has passed the value its submission signalled. The
	// largest such value is the newest, because the timeline is one counter across all of them.
	const uint64_t completed = p.fence->GetCompletedValue();
	uint32_t best = p.pictureSlot;
	uint64_t bestFence = 0;
	for (uint32_t i = 0; i < kSlots; ++i) {
		const uint64_t target = p.slotFence[i];
		if (target == 0 || completed == (std::numeric_limits<uint64_t>::max)() ||
			completed < target) {
			continue;
		}
		if (target > bestFence) {
			bestFence = target;
			best = i;
		}
	}
	bool ready = bestFence != 0;
	if (ready) {
		p.pictureSlot = best;
	}
	const uint32_t pictureSlot = p.pictureSlot;
	if (!ready) {
		// Nothing has retired yet, which is only true at the start. One bounded wait; if even that
		// finds nothing the picture stays the one already chosen rather than the frame being
		// dropped, because a dropped frame here alternates the effect on and off -- a strobe --
		// and a picture one generation older is the lesser fault.
		const uint64_t deadline = GetTickCount64() + 500;
		while (GetTickCount64() <= deadline) {
			const uint64_t now = p.fence->GetCompletedValue();
			for (uint32_t i = 0; i < kSlots; ++i) {
				if (p.slotFence[i] != 0 && now >= p.slotFence[i]) {
					p.pictureSlot = i;
					ready = true;
					break;
				}
			}
			if (ready) break;
			Sleep(0);
		}
		if (!ready) {
			if (++p.timeouts <= 3) {
				Logger::Get().Warn(fmt::format(
					"DLSSNR AMD: no slot retired within 500 ms (fence {} / newest {}); frame "
					"passed through ({} so far)",
					completed, p.slotFence[pictureSlot], p.timeouts));
			}
			p.resetHistory = true;
			return false;
		}
	}

	// The compositing goes on its own list and allocator. SubmitEngineJob closed and submitted
	// the one it had -- that is where the packet went -- and that submission is still in
	// flight, so this frame's output has to be recorded somewhere else entirely: recording
	// onto the closed list drops the calls, and reusing its allocator while the GPU still has
	// it fails outright.
	if (!p.outList) {
		Logger::Get().Error("DLSSNR AMD: no output list");
		p.failed = true;
		return true;
	}
	if (p.outFence != 0 && p.fence->GetCompletedValue() < p.outFence) {
		const uint64_t outWaitAt = Impl::Qpc();
		const uint64_t deadline = GetTickCount64() + 500;
		while (p.fence->GetCompletedValue() < p.outFence && GetTickCount64() < deadline) {
			Sleep(0);
		}
		p.compositeAccum[0] += Impl::Qpc() - outWaitAt;
	}
	if (FAILED(p.outAllocator->Reset()) ||
		FAILED(p.outList->Reset(p.outAllocator.get(), nullptr))) {
		Logger::Get().Error(fmt::format(
			"DLSSNR AMD: the output list could not be reset (fence {}/{})",
			p.fence->GetCompletedValue(), p.outFence));
		p.failed = true;
		return true;
	}
	auto* const list = p.outList.get();
	auto* const allocator = p.outAllocator.get();
	(void)allocator;

	// Point the member every helper records through at the output list, for the length of this
	// stage.
	//
	// Bind, DispatchSized, BindResolve and RunTemporal all record onto the member `list`, while
	// the barriers here take the local one -- and the member still named the engine's list, which
	// SubmitEngineJob had already closed and submitted. Recording onto a closed list drops the
	// calls without an error, so the temporal pass, the resolve and the output convert did
	// nothing at all: the resolve's surface came back empty and the screen stayed black, while
	// the input chain, recorded earlier when the list was still open, was perfectly fine. That
	// asymmetry is what made this look like a fault in the resolve rather than in the plumbing.
	struct EngineListScope {
		winrt::com_ptr<ID3D12GraphicsCommandList>& member;
		winrt::com_ptr<ID3D12GraphicsCommandList> saved;
		EngineListScope(winrt::com_ptr<ID3D12GraphicsCommandList>& m,
			const winrt::com_ptr<ID3D12GraphicsCommandList>& to) noexcept
			: member(m), saved(m) {
			member = to;
		}
		~EngineListScope() { member = saved; }
		EngineListScope(const EngineListScope&) = delete;
		EngineListScope& operator=(const EngineListScope&) = delete;
	} engineListScope(p.list, p.outList);

	ID3D12Resource* picture = p.net[pictureSlot].get();
	if (p.scaled) {
		// The residual for this frame, before anything composites it. With anti-flicker off
		// this only subtracts; with it on, earlier frames are blended in here.
		// Only when the picture is a generation this pass has not seen. With four slots the
		// newest retired one usually stands still for two or three frames while the engine
		// finishes the next, and an accumulator that blends one frame into the next would then
		// count the same generation once per frame instead of once -- the carried edit would
		// grow a little on every repeat and settle only by accident. Skipping leaves the last
		// residual in place, which is what the resolve reads.
		if (p.pictureSlot != p.temporalSourceSlot) {
			const uint64_t temporalAt = Impl::Qpc();
			if (!p.RunTemporal(context)) {
				Logger::Get().Error("DLSSNR AMD: the temporal pass stopped");
				p.failed = true;
				return true;
			}
			p.compositeAccum[1] += Impl::Qpc() - temporalAt;
			p.temporalSourceSlot = p.pictureSlot;
		}
		Barrier(list, p.resolved.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		// The base the edit is composited onto is THIS frame's picture, not the generation the
		// residual came from.
		//
		// The asynchronous handshake means the residual is a frame or two old -- that is what the
		// runtime means by "residual from an earlier frame" -- and what is old is the residual,
		// not the image. Building the output from the retired generation's own full-resolution
		// frame instead made the whole displayed picture that old: the content then advanced once
		// per engine job, about every 51 ms here, inside a presentation three times faster, which
		// reads as the picture stepping rather than moving.
		p.BindResolve(8, p.full[p.slot].get(), p.resolved.get(), p.baseline[pictureSlot].get(),
			p.historyResidual[p.historyIndex].get());
		// NR Intensity scales the edit the resolve applies -- the same idea as NGX's
		// residual intensity, and what a strength control on this effect should do. The
		// bound is the cap; the multiplier is what the slider moves.
		const uint32_t boundMilli = static_cast<uint32_t>(std::clamp(
			float(p.editBoundMilli) * std::clamp(p.settings.intensity, 0.0f, 2.0f),
			0.0f, 1000.0f));
		const UINT dims[5]{ p.width, p.height, p.netWidth, p.netHeight, boundMilli };
		const uint64_t resolveAt = Impl::Qpc();
		p.DispatchSized(p.resolve.get(), 8, p.width, p.height, 10, dims, 5);
		p.compositeAccum[2] += Impl::Qpc() - resolveAt;
		Barrier(list, p.resolved.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kStateShaderRead);
		picture = p.resolved.get();
	}

	if (p.outputIsFp16) {
		Barrier(list, picture, kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(list, p.sharedOut12.get(), kCommon,
			D3D12_RESOURCE_STATE_COPY_DEST);
		list->CopyResource(p.sharedOut12.get(), picture);
		Barrier(list, p.sharedOut12.get(), D3D12_RESOURCE_STATE_COPY_DEST, kCommon);
	} else {
		Barrier(list, p.sharedOut12.get(), kCommon,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		p.Bind(6, picture, DXGI_FORMAT_R16G16B16A16_FLOAT,
			p.sharedOut12.get(), p.outputFormat);
		// Handed to the dispatch rather than recorded before it, for the reason spelled out on
		// the reconstruction's own copy of this pass. It read the right value here only
		// because this list happens to have a root signature by now.
		const UINT outSettings[1]{ p.shoulderMilli };
		p.DispatchSized(p.srgbInput ? p.convertOutSrgb.get() : p.convertOut.get(), 6,
			p.width, p.height, 0, outSettings, 1);
		Barrier(list, p.sharedOut12.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kCommon);
	}
	const HRESULT closed = p.outList->Close();
	if (FAILED(closed)) {
		Logger::Get().Error(fmt::format(
			"DLSSNR AMD: the output list would not close (0x{:08x})",
			static_cast<uint32_t>(closed)));
		p.failed = true;
		return true;
	}
	ID3D12CommandList* lists[] = { p.outList.get() };
	const uint64_t submitAt = Impl::Qpc();
	p.SubmitTo(p.queue.get(), 1, lists);
	p.outFence = ++p.fenceValue;
	p.queue->Signal(p.fence.get(), p.outFence);
	p.compositeAccum[3] += Impl::Qpc() - submitAt;

	// Drained before the shared texture is read back on the other device. The copy below is
	// D3D11 work against a surface this queue is producing: without this wait it reads a
	// surface that has not been written yet, and the picture comes out black while every
	// stage in the chain reports success. The bypass path in this same function has always
	// waited here; the real one lost that wait when the output moved onto its own list.
	if (p.fence->GetCompletedValue() < p.outFence) {
		p.fence->SetEventOnCompletion(p.outFence, p.fenceEvent.get());
		if (WaitForSingleObject(p.fenceEvent.get(), 1000) != WAIT_OBJECT_0) {
			Logger::Get().Error("DLSSNR AMD: the output did not retire before the readback");
			p.failed = true;
			return true;
		}
	}

	// The other device is told where this queue has got to before it reads the shared surface.
	// Our own fence only says the work is finished here; this is what makes it visible there.
	// Its own fence, not the input's: signalling one object from both directions is what let
	// this wait resolve against a value that meant nothing here.
	if (p.context11x && p.outFence11 && p.outFence12) {
		const uint64_t crossing = ++p.outFenceValue;
		p.queue->Signal(p.outFence12.get(), crossing);
		if (FAILED(p.context11x->Wait(p.outFence11.get(), crossing))) {
			Logger::Get().Error("DLSSNR AMD: the other device refused the output crossing");
			p.failed = true;
			return true;
		}
	}

	// PROFILE: every stage of the chain, measured once, after the submissions for this frame
	// have all been made. The point is to see WHERE the value is lost rather than to guess:
	// each line is one stage, and the first one that reads zero is the one that broke it.
	if (!p.profiled && p.framesSeen >= 40) {
		p.profiled = true;
		// Each surface is measured from the state it is actually resting in: the two shared ones
		// are put back to COMMON at the end of every frame, the engine's own are left
		// shader-readable. Declaring the wrong one makes the transition an invalid barrier, and
		// an invalid barrier is not a failed read -- it is a copy against undefined contents,
		// which reads as zero and looks exactly like a chain that produced nothing.
		const auto mean = [&](ID3D12Resource* res, DXGI_FORMAT fmt, D3D12_RESOURCE_STATES before) {
			return res ? p.Measure(res, fmt, before) : -1.0f;
		};
		p.CrossTest();

			Logger::Get().Info(fmt::format(
			"DLSSNR AMD CONTROL: exposureTex (known 1.0) reads {:.4f} -- if this is 0 the "
			"measure is what is broken, not the chain",
			p.exposureTexture ? p.Measure(p.exposureTexture.get(),
				DXGI_FORMAT_R32_FLOAT, kStateShaderRead) : -1.0f));
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD ADAPTERS: d11Luid={:08x}:{:08x} d12Luid={:08x}:{:08x} same={}",
			static_cast<uint32_t>(p.luid11.HighPart), static_cast<uint32_t>(p.luid11.LowPart),
			static_cast<uint32_t>(p.luid12.HighPart), static_cast<uint32_t>(p.luid12.LowPart),
			std::memcmp(&p.luid11, &p.luid12, sizeof(LUID)) == 0));
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD HANDSHAKE: inSignal={} in12Completed={} outSignal={} out11Completed={} "
			"d3d12Fence={} inputIsFp16={} inputFmt={}",
			p.inFenceValue,
			p.inFence12 ? p.inFence12->GetCompletedValue() : 0,
			p.outFenceValue,
			p.outFence11 ? p.outFence11->GetCompletedValue() : 0,
			p.fence->GetCompletedValue(),
			p.inputIsFp16, static_cast<uint32_t>(p.inputFormat)));
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD PROFILE @frame {}: in11 {:.4f} | in12 {:.4f} | full {:.4f} | net {:.4f} | "
			"baseline {:.4f} | out11 {:.4f} | out12 {:.4f}",
			p.framesSeen,
			p.MeanOfOutput(p.sharedIn11.get()),
			mean(p.sharedIn12.get(), p.inputFormat, kCommon),
			mean(p.full[pictureSlot].get(), DXGI_FORMAT_R16G16B16A16_FLOAT, kStateShaderRead),
			mean(p.net[pictureSlot].get(), DXGI_FORMAT_R16G16B16A16_FLOAT, kStateShaderRead),
			mean(p.baseline[pictureSlot].get(), DXGI_FORMAT_R16G16B16A16_FLOAT,
				kStateShaderRead),
			p.MeanOfOutput(p.sharedOut11.get()),
			mean(p.sharedOut12.get(), p.outputFormat, kCommon)));
		// What the output stage was told to do, so a zero there can be told apart from a zero
		// produced by it: the scale, the branch the convert takes, and the two extents. The
		// resolve's own surface goes with them, because that is what the convert is handed.
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD OUTPUT @frame {}: pictureSlot {} scaled {} inputIsFp16 {} "
			"inputFmt {} outputFmt {} capture {}x{} network {}x{} resolved {:.4f}",
			p.framesSeen, pictureSlot, p.scaled, p.inputIsFp16,
			static_cast<uint32_t>(p.inputFormat), static_cast<uint32_t>(p.outputFormat),
			p.width, p.height, p.netWidth, p.netHeight,
			mean(p.resolved.get(), DXGI_FORMAT_R16G16B16A16_FLOAT, kStateShaderRead)));
	}

	// ONE-OFF PROBE, on the D3D11 side only.
	//
	// The D3D12-side measure shares the main command list and allocator with the stages that
	// are still being recorded, so its readback barrier and copy land in the middle of a
	// pipeline whose state contract it does not honour -- the numbers it produced were not
	// evidence. These three surfaces are all reachable from D3D11, which owns its own staging
	// texture and map, so what it reports is the actual contents.
	if (!p.probed && p.framesSeen >= 40) {
		p.probed = true;
		const float input = p.MeanOfOutput(p.sharedIn11.get());
		const float output = p.MeanOfOutput(p.sharedOut11.get());
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD PROBE (D3D11 side): magpie-input {:.4f} | shared-output {:.4f}",
			input, output));
	}

	// ONE-OFF: what each end of the chain actually contains, once, after the queue drains.
	if (p.framesSeen == 40) {
		p.fence->SetEventOnCompletion(p.outFence, p.fenceEvent.get());
		WaitForSingleObject(p.fenceEvent.get(), 2000);
		const float netMean = p.Measure(p.net[pictureSlot].get(),
			DXGI_FORMAT_R16G16B16A16_FLOAT, kStateShaderRead);
		const float fullMean = p.Measure(p.full[pictureSlot].get(),
			DXGI_FORMAT_R16G16B16A16_FLOAT, kStateShaderRead);
		const float outMean = p.Measure(p.sharedOut12.get(),
			p.outputFormat, D3D12_RESOURCE_STATE_COMMON);
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD PROBE: engine-output(net) {:.4f} | original(full) {:.4f} | final(sharedOut) {:.4f}",
			netMean, fullMean, outMean));
	}

	const uint64_t signal2 = ++p.fenceValue;
	p.queue->Signal(p.fence.get(), signal2);
	const uint64_t drainAt = Impl::Qpc();
	if (p.fence->GetCompletedValue() < signal2) {
		p.fence->SetEventOnCompletion(signal2, p.fenceEvent.get());
		if (WaitForSingleObject(p.fenceEvent.get(), 1000) != WAIT_OBJECT_0) {
			p.failed = true;
			return true;
		}
	}
	p.compositeAccum[4] += Impl::Qpc() - drainAt;
	phaseAt[5] = Impl::Qpc();
	// The frame's last signal: the guidance textures may be reused by the producer once this
	// value is reached, so the interop is told it before the next frame can Update.
	if (p.guidanceInterop) {
		p.guidanceInterop->MarkSubmitted(signal2);
	}

	p.context11->CopyResource(context.output, p.sharedOut11.get());

	// ---- where the frame went ----
	// Averaged over a window and printed at the optical-flow provider's cadence, so the two
	// can be read side by side. The stages are the ones the frame actually has: getting
	// Magpie's picture into the form the engine takes, resampling the motion guide to the
	// network's extent, recording and handing over the packet, waiting for the engine's
	// worker, compositing the result back, and the copy into Magpie's output texture.
	phaseAt[6] = Impl::Qpc();
	if (p.phaseWindowStart == 0) {
		p.phaseWindowStart = phaseAt[0];
	}
	for (int i = 0; i < 6; ++i) {
		p.phaseAccum[i] += phaseAt[i + 1] - phaseAt[i];
		p.prevFramePhasesMs[i] = static_cast<float>(
			double(phaseAt[i + 1] - phaseAt[i]) * Impl::MsPerTick());
	}
	p.phaseAccum[6] += phaseAt[6] - phaseAt[0];
	p.prevFrameId = context.frameId;
	p.prevFrameJob = At<UINT>(p.runtime, p.rva->jobCounter);
	p.prevFrameTotalMs = static_cast<float>(
		double(phaseAt[6] - phaseAt[0]) * Impl::MsPerTick());
	if (++p.phaseFrames >= 120) {
		const double msPerFrame = Impl::MsPerTick() / double(p.phaseFrames);
		const double span = double(phaseAt[6] - p.phaseWindowStart) * Impl::MsPerTick();
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD phases over {} frames, {:.2f} ms/frame: convert {:.2f} + motion "
			"{:.2f} + record {:.2f} + engine {:.2f} + composite {:.2f} + copy {:.2f} "
			"+ rest {:.2f} = {:.2f} ms",
			p.phaseFrames, span / double(p.phaseFrames),
			p.phaseAccum[0] * msPerFrame, p.phaseAccum[1] * msPerFrame,
			p.phaseAccum[2] * msPerFrame, p.phaseAccum[3] * msPerFrame,
			p.phaseAccum[4] * msPerFrame, p.phaseAccum[5] * msPerFrame,
			std::max(0.0, (span - double(p.phaseAccum[6]) * msPerFrame)
				/ double(p.phaseFrames)),
			p.phaseAccum[6] * msPerFrame));
		// What the composite stage is made of. Two of the five are blocking waits on the
		// CPU, and the stage total cannot say which.
		const double compTotal = double(p.compositeAccum[0] + p.compositeAccum[1] +
			p.compositeAccum[2] + p.compositeAccum[3] + p.compositeAccum[4]) * msPerFrame;
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD composite over {} frames: out-wait {:.2f} + temporal {:.2f} + "
			"resolve {:.2f} + submit {:.2f} + drain {:.2f} = {:.2f} ms/frame",
			p.phaseFrames,
			double(p.compositeAccum[0]) * msPerFrame, double(p.compositeAccum[1]) * msPerFrame,
			double(p.compositeAccum[2]) * msPerFrame, double(p.compositeAccum[3]) * msPerFrame,
			double(p.compositeAccum[4]) * msPerFrame, compTotal));
		for (int i = 0; i < 7; ++i) {
			p.phaseAccum[i] = 0;
		}
		for (int i = 0; i < 5; ++i) {
			p.compositeAccum[i] = 0;
		}
		p.phaseFrames = 0;
		p.phaseWindowStart = phaseAt[6];

		if (p.frameIntervalCount) {
			std::vector<float> sorted(
				p.frameIntervalsMs, p.frameIntervalsMs + p.frameIntervalCount);
			std::sort(sorted.begin(), sorted.end());
			const auto at = [&sorted](double q) noexcept {
				return sorted[static_cast<size_t>(q * double(sorted.size() - 1))];
			};
			uint32_t over33 = 0, over50 = 0;
			for (uint32_t i = 0; i < p.frameIntervalCount; ++i) {
				over33 += p.frameIntervalsMs[i] > 33.3f ? 1u : 0u;
				over50 += p.frameIntervalsMs[i] > 50.0f ? 1u : 0u;
			}
			Logger::Get().Info(fmt::format(
				"DLSSNR AMD frame time over {} frames: p50 {:.2f} p95 {:.2f} p99 {:.2f} "
				"max {:.2f} ms | {} over 33.3 ms, {} over 50",
				p.frameIntervalCount, at(0.50), at(0.95), at(0.99), sorted.back(),
				over33, over50));
			p.frameIntervalCount = 0;
		}
	}
	return true;
}

// Magpie calls this before it lets go of the surfaces and before another session may start,
// and it has to be a real quiescence point. Two things make the stub that used to be here
// unsafe, and both end in a removed device rather than in a lost frame.
//
// The engine runs the network on its own worker, so a fence on this queue says nothing about
// whether a kernel is still writing into a shared surface. And `~Impl` explains why the runtime
// is never unloaded: it starts threads holding references into its own image. So anything this
// backend leaves running keeps running into the next session, on the same runtime instance,
// while the surfaces it was reading and writing have already been released. The symptom is
// DXGI_ERROR_DEVICE_REMOVED (0x887A0005) on the first frames of the *second* session -- start,
// stop, start -- which is exactly the sequence that was reported as reproducible.
//
// What this cannot do is stop the runtime driving itself: its own setup thread installs the
// detours that put its work on the game's queue, and that thread is not ours to join. Tools
// patch_runtime_050.py neutralises the call that starts it, which is the other half.
bool DlssnrAmdBackend::Drain() noexcept {
	auto& p = *_impl;
	if (!p.ready || !p.runtime || !p.rva) {
		return true;
	}

	// The engine's half. The highest job any slot was given is the last one to wait for;
	// syncCounter is the counter the worker advances when it publishes a result, and it is the
	// same quantity WaitForEngine polls inside the frame loop.
	uint32_t lastJob = 0;
	for (uint32_t k = 0; k < kSlots; ++k) {
		if (p.slotJob[k] > lastJob) {
			lastJob = p.slotJob[k];
		}
	}
	bool quiesced = true;
	if (lastJob && At<UINT>(p.runtime, p.rva->syncCounter) < lastJob) {
		if (!p.WaitForEngine(lastJob, kDrainDeadlineMs)) {
			// A frame that arrives late is already lost, so this is not fatal on its own --
			// but it does mean the wait below is the only thing standing between a running
			// kernel and a released surface, and the caller should know it happened.
			Logger::Get().Warn(fmt::format(
				"DLSSNR AMD: the engine did not publish job {} within {} ms while draining",
				lastJob, kDrainDeadlineMs));
			quiesced = false;
		}
	}

	// This queue's half. The signal is taken after everything already submitted, so waiting on
	// it is waiting for all of it.
	if (p.fence) {
		const uint64_t signal = ++p.fenceValue;
		p.queue->Signal(p.fence.get(), signal);
		if (p.fence->GetCompletedValue() < signal) {
			p.fence->SetEventOnCompletion(signal, p.fenceEvent.get());
			if (WaitForSingleObject(p.fenceEvent.get(), static_cast<DWORD>(kDrainDeadlineMs)) != WAIT_OBJECT_0) {
				Logger::Get().Warn("DLSSNR AMD: this queue did not retire its work while draining");
				quiesced = false;
			}
		}
	}

	// Now that both sides are idle, hand the slots back. RetireSubmission is the only thing
	// that turns a recorded slot into a reusable one, and leaving them recorded would make the
	// next session wait on fences that belong to the one being torn down.
	p.RetireSubmission("drain");
	return quiesced;
}

}
