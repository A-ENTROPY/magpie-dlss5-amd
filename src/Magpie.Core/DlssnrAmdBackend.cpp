#include "pch.h"
#include "DlssnrAmdBackend.h"
#include "FrameGuidanceD3D12Interop.h"
#include "DeviceResources.h"
#include "Logger.h"
#include "Win32Helper.h"

// d3d12.h is not part of the precompiled header. DLSSNRFilter.cpp gets its D3D12 types
// through the NGX headers, which this backend deliberately does not include -- it has no
// business pulling NVIDIA's SDK into a path that runs on a Radeon.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <bcrypt.h>
#include <d3dcompiler.h>

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
//   base + 0x12380   bool(void* ctx, const std::string* weights)   engine init
//   base + 0xa0b0    void(Packet*)                                 record a job
//   base + 0x4640    void(queue, n, lists)                         notify submitted
//   base + 0x764d8   data                                          the ctx struct
// ---------------------------------------------------------------------------
constexpr uintptr_t kRvaDevice = 0x764c8;
constexpr uintptr_t kRvaQueue = 0x764d0;
constexpr uintptr_t kRvaInitCtx = 0x764d8;
constexpr uintptr_t kRvaHipDevice = 0x76f20;
constexpr uintptr_t kRvaInlineMode = 0x76be0;
constexpr uintptr_t kRvaInterop = 0x76c8c;
constexpr uintptr_t kRvaEnabled = 0x76e1c;
constexpr uintptr_t kRvaUseFsrInputs = 0x76e1e;
constexpr uintptr_t kRvaUseDepth = 0x76e1f;
constexpr uintptr_t kRvaTonemap = 0x76e20;
constexpr uintptr_t kRvaFlag767f8 = 0x767f8;
constexpr uintptr_t kRvaHistory = 0x765f0;
constexpr uintptr_t kRvaWantHistory = 0x765f8;
constexpr uintptr_t kRvaPerPassFlag = 0x76e1d;
constexpr uintptr_t kRvaDepthInverted = 0x76e10;
constexpr uintptr_t kRvaDepthExplicit = 0x76e14;
constexpr uintptr_t kRvaLocalTone = 0x76e30;
constexpr uintptr_t kRvaLocalStructure = 0x76e34;
constexpr uintptr_t kRvaSkinStructure = 0x76e38;
constexpr uintptr_t kRvaCharMask = 0x76e40;
constexpr uintptr_t kRvaToneChannels = 0x76e44;
constexpr uintptr_t kRvaJobCounter = 0x76d74;
constexpr uintptr_t kRvaStatusFlag = 0x767fa;
constexpr uintptr_t kRvaSyncCounter = 0x76c14;

constexpr uintptr_t kRvaInit = 0x12380;
constexpr uintptr_t kRvaRecord = 0xa0b0;
constexpr uintptr_t kRvaNotify = 0x4640;

// sha256 of the build these offsets belong to. Anything else is refused rather than
// attempted: the offsets point into a different layout and the failure mode is a jump
// into nothing, not an error.
constexpr uint8_t kRuntimeSha256[32] = {
	0x3c, 0x9c, 0xa1, 0x3f, 0x0f, 0x5f, 0xc3, 0x6a, 0x69, 0x0b, 0xa4, 0x24, 0xc4, 0x57,
	0x00, 0x3b, 0xcf, 0xcc, 0x10, 0x80, 0xb4, 0xb7, 0x85, 0x97, 0x4c, 0xdd, 0x7e, 0x9a,
	0xe2, 0xbc, 0x1d, 0xd8
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
	UINT pad4c;
};
static_assert(sizeof(Packet) == 0x50, "Packet must be 0x50 bytes");
static_assert(offsetof(Packet, scaleX) == 0x44, "scaleX must sit at 0x44");

using InitFn = bool(__fastcall*)(void*, const std::string*);
using RecordFn = void(__fastcall*)(Packet*);
using NotifyFn = void(__fastcall*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
using HipSetFn = int (*)(int);

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
constexpr char kDownsampleShader[] = R"(
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
cbuffer Extent : register(b0) { uint w; uint h; uint sourceW; uint sourceH; };
[numthreads(8, 8, 1)]
void main(uint3 p : SV_DispatchThreadID)
{
	if (p.x >= w || p.y >= h) return;
	// Integrate the whole source footprint. A single bilinear sample loses narrow
	// emissive lines once the model runs well below the input resolution.
	float2 lo = float2(p.xy) * float2(sourceW, sourceH) / float2(w, h);
	float2 hi = float2(p.xy + 1) * float2(sourceW, sourceH) / float2(w, h);
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
	uint w; uint h; float weight; uint historyValid;
	uint useMotion; uint motionScaleXMilli; uint motionScaleYMilli;
};

float3 Guide(float3 c) { return c / (1 + abs(c)); }
float MaxAbs(float3 v) { return max(abs(v.r), max(abs(v.g), abs(v.b))); }

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

	float3 current = raw.rgb - base.rgb;
	float3 result = current;

	if (historyValid != 0 && weight > 0) {
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
			result = lerp(current, safe, weight * trust);
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



bool HashMatches(const std::filesystem::path& file) noexcept {
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
	return std::memcmp(digest, kRuntimeSha256, sizeof(digest)) == 0;
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

ID3D12Device* CreateDeviceOnAdapter(ID3D11Device* device11) noexcept {
	winrt::com_ptr<IDXGIDevice> dxgiDevice;
	if (FAILED(device11->QueryInterface(IID_PPV_ARGS(dxgiDevice.put())))) {
		return nullptr;
	}
	winrt::com_ptr<IDXGIAdapter> adapter;
	if (FAILED(dxgiDevice->GetAdapter(adapter.put()))) {
		return nullptr;
	}

	winrt::com_ptr<ID3D12Device> device12;
	if (FAILED(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_12_0,
		IID_PPV_ARGS(device12.put())))) {
		return nullptr;
	}
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
	winrt::com_ptr<ID3D12Fence> fence;

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

	// The engine's surfaces.
	// The engine works in place: the texture named in the packet is both what it reads
	// and what it writes. There is one surface here, not two.
	winrt::com_ptr<ID3D12Resource> net;
	// The frame as the engine will not see it: full resolution, and holding the version
	// from before the engine touched anything. `full` is what the resolve takes detail
	// from, `baseline` is what it subtracts to isolate the edit.
	winrt::com_ptr<ID3D12Resource> full;
	winrt::com_ptr<ID3D12Resource> baseline;
	winrt::com_ptr<ID3D12Resource> resolved;
	winrt::com_ptr<ID3D12Resource> motion;
	winrt::com_ptr<ID3D12Resource> depth;
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
	bool ready = false;
	bool failed = false;
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
	// The previous call's samples, for the flicker measure. Never used for anything else.
	std::vector<float> sample;
	std::vector<float> previousSample;
	// TEMPORARY DIAGNOSTIC: how many frames of the series have been logged.
	int probeSamples = 0;

	uint32_t width = 0, height = 0;

	~Impl() {
		// The runtime stays loaded for the life of the process: it starts worker threads
		// holding references into its own image, and unloading it under them is not
		// something this backend can make safe.
	}

	D3D12_CPU_DESCRIPTOR_HANDLE Cpu(uint32_t slot) const noexcept {
		D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
		h.ptr += SIZE_T(slot) * descriptorStride;
		return h;
	}

	bool LoadRuntime() noexcept;
	bool CreateHip() noexcept;
	bool CreatePipeline() noexcept;
	bool CreateSized(uint32_t w, uint32_t h, DXGI_FORMAT inFmt, DXGI_FORMAT outFmt) noexcept;
	bool InitEngine(const std::filesystem::path& weightsPath) noexcept;
	bool CreateExposure() noexcept;
	void Bind(uint32_t slot, ID3D12Resource* srv, DXGI_FORMAT srvFormat,
		ID3D12Resource* uav, DXGI_FORMAT uavFormat) noexcept;
	void BindResolve(uint32_t slot, ID3D12Resource* srv, ID3D12Resource* uav,
		ID3D12Resource* baselineTexture, ID3D12Resource* edited) noexcept;
	void BindTemporal(uint32_t slot, ID3D12Resource* edited, ID3D12Resource* nextResidual,
		ID3D12Resource* nextGuide, ID3D12Resource* baselineTexture,
		ID3D12Resource* historyResidualTexture,
		ID3D12Resource* historyGuideTexture, ID3D12Resource* motionTexture) noexcept;
	void CreateSrv(ID3D12Resource* res, D3D12_CPU_DESCRIPTOR_HANDLE where) noexcept;
	bool RunTemporal(const NativeEffectDrawContext& context) noexcept;
	void Dispatch(ID3D12PipelineState* pso, uint32_t slot) noexcept;
	// `residualTableSlot` is where the second descriptor table starts, or 0 for a pass that
	// has no second table. It is a slot rather than a flag because the passes lay their
	// descriptors out differently: the temporal pass writes two UAVs before its extra
	// sources, the resolve writes one.
	void DispatchSized(ID3D12PipelineState* pso, uint32_t slot, uint32_t dw, uint32_t dh,
		uint32_t residualTableSlot, const UINT* constants = nullptr,
		uint32_t constantCount = 4) noexcept;
	bool WaitForEngine(uint64_t deadlineMs) noexcept;
	void DestroySized() noexcept;
	float Measure(ID3D12Resource* res, DXGI_FORMAT format,
		D3D12_RESOURCE_STATES before, uint64_t* nonFinite = nullptr,
		float* meanAbsDelta = nullptr) noexcept;
};

bool DlssnrAmdBackend::Impl::LoadRuntime() noexcept {
	if (runtime) {
		return true;
	}

	const auto dir = ExeDirectory();
	const auto runtimePath = dir / kRuntimeName;
	const auto weightsPath = dir / kWeightsName;

	std::error_code ec;
	if (!std::filesystem::exists(runtimePath, ec) ||
		!std::filesystem::exists(weightsPath, ec)) {
		return false;
	}
	if (!HashMatches(runtimePath)) {
		Logger::Get().Error(
			"DLSSNR AMD: dlssnr_amd_pass1.dll is not the build these offsets belong to; "
			"refusing rather than guessing at offsets into a different layout");
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

	init = reinterpret_cast<InitFn>(reinterpret_cast<uintptr_t>(runtime) + kRvaInit);
	record = reinterpret_cast<RecordFn>(reinterpret_cast<uintptr_t>(runtime) + kRvaRecord);
	notify = reinterpret_cast<NotifyFn>(reinterpret_cast<uintptr_t>(runtime) + kRvaNotify);
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
		Logger::Get().Error("DLSSNR AMD: no HIP device matches the render adapter");
		return false;
	}
	if (hipSet(hipDevice) != 0) {
		return false;
	}
	return true;
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
	if (!CreateSharedTexture(device11, device12.get(), w, h, inFmt, false,
			sharedIn11, sharedIn12) ||
		!CreateSharedTexture(device11, device12.get(), w, h, outFmt, true,
			sharedOut11, sharedOut12)) {
		Logger::Get().Error("DLSSNR AMD: could not share the frame between the devices");
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
	if (!CreateEngineTexture(device12.get(), w, h,
			DXGI_FORMAT_R16G16B16A16_FLOAT, full)) {
		return false;
	}

	// The engine's surface. It works in place: the texture named in the packet is both what
	// it reads and what it writes, so there is one surface there, not two. The reference
	// hands over exactly one such pointer as `Packet::colour`.
	if (!CreateEngineTexture(device12.get(), netWidth, netHeight,
			DXGI_FORMAT_R16G16B16A16_FLOAT, net)) {
		return false;
	}
	if (scaled) {
		// The network's output for this frame before the engine ran, and where the resolve
		// writes. The resolve subtracts the first from what the engine produced to isolate
		// the edit, and needs `full` to put the untouched detail back underneath it.
		if (!CreateEngineTexture(device12.get(), netWidth, netHeight,
				DXGI_FORMAT_R16G16B16A16_FLOAT, baseline) ||
			!CreateEngineTexture(device12.get(), w, h,
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
	if (!CreateEngineTexture(device12.get(), netWidth, netHeight,
			DXGI_FORMAT_R16G16_FLOAT, motion) ||
		!CreateEngineTexture(device12.get(), netWidth, netHeight,
			DXGI_FORMAT_R32_FLOAT, depth)) {
		return false;
	}
	// Held steady rather than adapted; the reason and the measurements are on kExposureValue.
	if (!CreateExposure()) {
		return false;
	}
	return true;
}

void DlssnrAmdBackend::Impl::DestroySized() noexcept {
	sharedIn11 = nullptr;
	sharedIn12 = nullptr;
	sharedOut11 = nullptr;
	sharedOut12 = nullptr;
	net = nullptr;
	full = nullptr;
	baseline = nullptr;
	resolved = nullptr;
	for (int i = 0; i < 2; ++i) {
		historyResidual[i] = nullptr;
		historyGuide[i] = nullptr;
	}
	// A new extent means the history describes the wrong geometry.
	resetHistory = true;
	temporalValid = false;
	motion = nullptr;
	depth = nullptr;
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
	uint32_t slot, ID3D12Resource* srv, DXGI_FORMAT srvFormat,
	ID3D12Resource* uav, DXGI_FORMAT uavFormat
) noexcept {
	D3D12_SHADER_RESOURCE_VIEW_DESC s{};
	s.Format = srvFormat;
	s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	s.Texture2D.MipLevels = 1;
	device12->CreateShaderResourceView(srv, &s, Cpu(slot));

	D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
	u.Format = uavFormat;
	u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device12->CreateUnorderedAccessView(uav, nullptr, &u, Cpu(slot + 1));
}

void DlssnrAmdBackend::Impl::Dispatch(ID3D12PipelineState* pso, uint32_t slot) noexcept {
	DispatchSized(pso, slot, width, height, false);
}

void DlssnrAmdBackend::Impl::DispatchSized(ID3D12PipelineState* pso, uint32_t slot,
	uint32_t dw, uint32_t dh, uint32_t residualTableSlot, const UINT* constants,
	uint32_t constantCount
) noexcept {
	// `slot` matters more than it looks. Descriptors are read when the GPU executes the
	// list, not when it is recorded, so two stages sharing a slot both end up reading
	// whichever bind happened last. Each stage therefore gets its own set, and the root
	// table is pointed at that set here.
	ID3D12DescriptorHeap* h = heap.get();
	list->SetComputeRootSignature(root.get());
	list->SetDescriptorHeaps(1, &h);
	D3D12_GPU_DESCRIPTOR_HANDLE table = heap->GetGPUDescriptorHandleForHeapStart();
	table.ptr += UINT64(slot) * descriptorStride;
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
	// take it in the same four-slot shape the reference uses: destination width and height
	// followed by the source's.
	if (constants) {
		list->SetComputeRoot32BitConstants(1, constantCount, constants, 0);
	}
	list->Dispatch((dw + 7) / 8, (dh + 7) / 8, 1);
}

void DlssnrAmdBackend::Impl::BindResolve(uint32_t slot, ID3D12Resource* srv,
	ID3D12Resource* uav, ID3D12Resource* baselineTexture, ID3D12Resource* edited
) noexcept {
	// t0/u0 through the usual pair, then t1 and t2 sitting immediately after it so the
	// resolve's second root table finds them two descriptors along.
	Bind(slot, srv, DXGI_FORMAT_R16G16B16A16_FLOAT, uav, DXGI_FORMAT_R16G16B16A16_FLOAT);

	CreateSrv(baselineTexture, Cpu(slot + 2));
	CreateSrv(edited, Cpu(slot + 3));
}

void DlssnrAmdBackend::Impl::BindTemporal(uint32_t slot, ID3D12Resource* edited,
	ID3D12Resource* nextResidual, ID3D12Resource* nextGuide,
	ID3D12Resource* baselineTexture, ID3D12Resource* historyResidualTexture,
	ID3D12Resource* historyGuideTexture, ID3D12Resource* motionTexture
) noexcept {
	// t0 and u0 through the usual pair, then u1 immediately after it, and the four sources
	// from t1 in the table that follows -- which is why this pass's second table starts at
	// slot + 3 while the resolve's starts at slot + 2.
	Bind(slot, edited, DXGI_FORMAT_R16G16B16A16_FLOAT, nextResidual,
		DXGI_FORMAT_R16G16B16A16_FLOAT);

	D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
	u.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
	u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device12->CreateUnorderedAccessView(nextGuide, nullptr, &u, Cpu(slot + 2));

	CreateSrv(baselineTexture, Cpu(slot + 3));
	CreateSrv(historyResidualTexture, Cpu(slot + 4));
	CreateSrv(historyGuideTexture, Cpu(slot + 5));

	D3D12_SHADER_RESOURCE_VIEW_DESC m{};
	m.Format = DXGI_FORMAT_R16G16_FLOAT;
	m.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	m.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	m.Texture2D.MipLevels = 1;
	device12->CreateShaderResourceView(motionTexture, &m, Cpu(slot + 6));
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
	BindTemporal(12, net.get(), historyResidual[to].get(), historyGuide[to].get(),
		baseline.get(), historyResidual[from].get(), historyGuide[from].get(),
		motion.get());
	// Modes two and above reproject the history with the flow; one is static accumulation.
	const uint32_t useMotion =
		antiFlickerMode >= 2 && motionReady ? 1u : 0u;
	const UINT constants[7]{ netWidth, netHeight, weightMilli, hasHistory ? 1u : 0u,
		useMotion,
		uint32_t(motionScaleX * 1000.0f), uint32_t(motionScaleY * 1000.0f) };
	DispatchSized(temporal.get(), 12, netWidth, netHeight, 15, constants, 7);
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
	At<ID3D12Device*>(runtime, kRvaDevice) = device12.get();
	device12.get()->AddRef();
	At<ID3D12CommandQueue*>(runtime, kRvaQueue) = queue.get();
	queue.get()->AddRef();
	At<int>(runtime, kRvaHipDevice) = hipDevice;
	At<uint8_t>(runtime, kRvaInlineMode) = 1;
	At<uint8_t>(runtime, kRvaEnabled) = 1;
	// Interop is the runtime's own key and is left to dlssnr_on_amd.ini, the way Tonemap is;
	// setting it here would override whatever the file says. UseFsrInputs and UseDepth are
	// written per pass from the effect's parameters instead.
	//
	// Inline stays pinned, and it is the one engine key this backend does not expose. The
	// engine in inline mode completes the job on the frame it was given, which is what the
	// rest of this backend assumes when it converts the result out; the asynchronous mode
	// hands back the previous frame instead and the chain has no such path.

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
	// could be tested -- and the result was a black frame, so they are pinned again. Do not
	// unpin them without testing one variable at a time.
	At<uint8_t>(runtime, kRvaInlineMode) = 1;
	At<uint8_t>(runtime, kRvaInterop) = 1;

	const std::string weights = weightsPath.string();
	if (!CallInit(init,
		reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(runtime) + kRvaInitCtx),
		&weights)) {
		Logger::Get().Error("DLSSNR AMD: engine initialisation failed");
		return false;
	}

	At<uint8_t>(runtime, kRvaFlag767f8) = 1;
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
	queue->ExecuteCommandLists(1, lists);
	const uint64_t signal = ++fenceValue;
	queue->Signal(fence.get(), signal);
	if (fence->GetCompletedValue() < signal) {
		fence->SetEventOnCompletion(signal, fenceEvent.get());
		if (WaitForSingleObject(fenceEvent.get(), 1000) != WAIT_OBJECT_0) {
			return false;
		}
	}
	return true;
}

bool DlssnrAmdBackend::Impl::WaitForEngine(uint64_t deadlineMs) noexcept {
	// The network runs on the engine's own worker, so a fence on this queue says nothing
	// about whether a result exists. The engine publishes its progress in its sync
	// counter, and that is what the working implementation polls.
	const uint32_t wanted = At<UINT>(runtime, kRvaJobCounter);
	const uint64_t deadline = GetTickCount64() + deadlineMs;
	while (At<UINT>(runtime, kRvaSyncCounter) < wanted) {
		if (GetTickCount64() > deadline) {
			return false;
		}
		Sleep(0);
	}
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
		"style={} intensity={:.2f}",
		settings.localToneStrength, settings.localStructureStrength,
		settings.skinStructureStrength, settings.useAutoMask ? "on" : "off",
		settings.amdTemporal, settings.amdToneChannels, settings.amdUseDepth,
		settings.amdUseFsrInputs,
		settings.enableInputResolutionScaling ? int(settings.inputResolutionPercent) : 100,
		settings.style, settings.intensity));
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
	p.modelScale = settings.enableInputResolutionScaling
		? float(std::clamp<uint32_t>(settings.inputResolutionPercent, 25, 100)) / 100.0f
		: 1.0f;

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

	p.device12.attach(CreateDeviceOnAdapter(p.device11));
	if (!p.device12) {
		Logger::Get().Error("DLSSNR AMD: no D3D12 device on the render adapter");
		return false;
	}

	D3D12_COMMAND_QUEUE_DESC qd{};
	qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	if (FAILED(p.device12->CreateCommandQueue(&qd, IID_PPV_ARGS(p.queue.put()))) ||
		FAILED(p.device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
			IID_PPV_ARGS(p.allocator.put()))) ||
		FAILED(p.device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
			p.allocator.get(), nullptr, IID_PPV_ARGS(p.list.put()))) ||
		FAILED(p.device12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
			IID_PPV_ARGS(p.fence.put())))) {
		Logger::Get().Error("DLSSNR AMD: could not create the D3D12 command objects");
		return false;
	}
	p.fenceEvent.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
	if (!p.fenceEvent.valid() || FAILED(p.list->Close())) {
		return false;
	}

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
	p.guidanceInterop = std::make_unique<FrameGuidanceD3D12Interop>();
	if (!p.guidanceInterop->Initialize(p.device12.get(), p.fence.get())) {
		Logger::Get().Error("DLSSNR AMD: guidance interop failed to initialise");
		return false;
	}

	if (!p.LoadRuntime() || !p.CreateHip() || !p.CreatePipeline()) {
		return false;
	}
	if (!p.CreateSized(inputDesc.Width, inputDesc.Height,
			inputDesc.Format, outputDesc.Format)) {
		return false;
	}
	if (!p.InitEngine(ExeDirectory() / kWeightsName)) {
		return false;
	}

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
	return p.CreateSized(inDesc.Width, inDesc.Height, inDesc.Format, outDesc.Format);
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

	if (!probe || probeBytes < total) {
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

	if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.get(), nullptr))) {
		return -1.0f;
	}
	Barrier(list.get(), res, before, D3D12_RESOURCE_STATE_COPY_SOURCE);

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
	list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

	Barrier(list.get(), res, D3D12_RESOURCE_STATE_COPY_SOURCE, before);
	if (FAILED(list->Close())) {
		return -1.0f;
	}
	ID3D12CommandList* lists[] = { list.get() };
	queue->ExecuteCommandLists(1, lists);
	const uint64_t sig = ++fenceValue;
	queue->Signal(fence.get(), sig);
	if (fence->GetCompletedValue() < sig) {
		fence->SetEventOnCompletion(sig, fenceEvent.get());
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

	// A texture shared between two devices sits in COMMON on the D3D12 side and has to
	// go back there before the other device may touch it again, so every crossing below
	// is COMMON -> in use -> COMMON.
	p.context11->CopyResource(p.sharedIn11.get(), context.input);
	p.context11->Flush();

	if (FAILED(p.allocator->Reset()) || FAILED(p.list->Reset(p.allocator.get(), nullptr))) {
		return false;
	}

	// ---- 1. Magpie's frame into a full-resolution RGBA16F surface ----
	// The engine's format is RGBA16F and Magpie's usually is not, so this is where the two
	// are reconciled. The result is kept at full size whether or not the network will see
	// it: at a reduced scale the resolve needs the untouched full-resolution frame to put
	// its detail back.
	if (p.inputIsFp16) {
		Barrier(p.list.get(), p.sharedIn12.get(), kCommon, D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(p.list.get(), p.full.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_DEST);
		p.list->CopyResource(p.full.get(), p.sharedIn12.get());
		Barrier(p.list.get(), p.sharedIn12.get(), D3D12_RESOURCE_STATE_COPY_SOURCE, kCommon);
		Barrier(p.list.get(), p.full.get(), D3D12_RESOURCE_STATE_COPY_DEST,
			kStateShaderRead);
	} else {
		Barrier(p.list.get(), p.sharedIn12.get(), kCommon, kStateShaderRead);
		Barrier(p.list.get(), p.full.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		p.Bind(2, p.sharedIn12.get(), p.inputFormat, p.full.get(),
			DXGI_FORMAT_R16G16B16A16_FLOAT);
		p.Dispatch(p.srgbInput ? p.convertInSrgb.get() : p.convertIn.get(), 2);
		Barrier(p.list.get(), p.sharedIn12.get(), kStateShaderRead, kCommon);
		Barrier(p.list.get(), p.full.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kStateShaderRead);
	}

	// ---- 1b. the frame the network actually sees ----
	// At a reduced scale this is an area average rather than a bilinear sample, because a
	// point sample loses narrow bright features once the model runs well below the input --
	// the reference makes the same choice for the same reason. The copy into `baseline` has
	// to happen before the engine runs, since the engine edits its surface in place and the
	// resolve needs the difference.
	if (p.downscaling) {
		Barrier(p.list.get(), p.net.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		p.Bind(4, p.full.get(), DXGI_FORMAT_R16G16B16A16_FLOAT, p.net.get(),
			DXGI_FORMAT_R16G16B16A16_FLOAT);
		const UINT dims[4]{ p.netWidth, p.netHeight, p.width, p.height };
		p.DispatchSized(p.downsample.get(), 4, p.netWidth, p.netHeight, 0, dims);
		Barrier(p.list.get(), p.net.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kStateShaderRead);
	} else if (p.scaled) {
		Barrier(p.list.get(), p.full.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(p.list.get(), p.net.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_DEST);
		p.list->CopyResource(p.net.get(), p.full.get());
		Barrier(p.list.get(), p.net.get(), D3D12_RESOURCE_STATE_COPY_DEST,
			kStateShaderRead);
		Barrier(p.list.get(), p.full.get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
			kStateShaderRead);
	}
	if (p.scaled) {
		Barrier(p.list.get(), p.net.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(p.list.get(), p.baseline.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_DEST);
		p.list->CopyResource(p.baseline.get(), p.net.get());
		Barrier(p.list.get(), p.baseline.get(), D3D12_RESOURCE_STATE_COPY_DEST,
			kStateShaderRead);
		Barrier(p.list.get(), p.net.get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
			kStateShaderRead);
	}

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
			Barrier(p.list.get(), p.motion.get(), kStateShaderRead,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			p.Bind(20, p.guidanceInterop->Motion(), DXGI_FORMAT_R16G16_FLOAT,
				p.motion.get(), DXGI_FORMAT_R16G16_FLOAT);
			const UINT dims[4]{ p.netWidth, p.netHeight, sourceW, sourceH };
			p.DispatchSized(p.motionResample.get(), 20, p.netWidth, p.netHeight, 0, dims);
			Barrier(p.list.get(), p.motion.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
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
		p.Bind(6, p.full.get(), DXGI_FORMAT_R16G16B16A16_FLOAT,
			p.sharedOut12.get(), p.outputFormat);
		p.list->SetComputeRoot32BitConstants(1, 1, &p.shoulderMilli, 0);
		p.Dispatch(p.srgbInput ? p.convertOutSrgb.get() : p.convertOut.get(), 6);
		Barrier(p.list.get(), p.sharedOut12.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kCommon);
		p.list->Close();
		ID3D12CommandList* bl[] = { p.list.get() };
		p.queue->ExecuteCommandLists(1, bl);
		const uint64_t bs = ++p.fenceValue;
		p.queue->Signal(p.fence.get(), bs);
		if (p.fence->GetCompletedValue() < bs) {
			p.fence->SetEventOnCompletion(bs, p.fenceEvent.get());
			WaitForSingleObject(p.fenceEvent.get(), 1000);
		}
		p.context11->CopyResource(context.output, p.sharedOut11.get());

		if (!p.probed && ++p.framesSeen >= 10) {
			p.probed = true;
			const float inMean = p.Measure(p.sharedIn12.get(), p.inputFormat,
				D3D12_RESOURCE_STATE_COMMON);
			const float netMean = p.Measure(p.full.get(),
				DXGI_FORMAT_R16G16B16A16_FLOAT, kStateShaderRead);
			const float outMean = p.Measure(p.sharedOut12.get(), p.outputFormat,
				D3D12_RESOURCE_STATE_COMMON);
			Logger::Get().Info(fmt::format(
				"DLSSNR AMD probe (bypassed): frame in {:.4f}, converted {:.4f}, "
				"converted out {:.4f}", inMean, netMean, outMean));
		}
		return true;
	}

	// ---- 2. the engine's own state, immediately before it records ----
	// Not set-and-forget: the working implementation rewrites the whole block per pass.
	// 0x76e1d is the engine's `Temporal` key as well as the per-pass flag the reference
	// asserts; the control drives it directly.
	At<uint8_t>(p.runtime, kRvaPerPassFlag) = p.settings.amdTemporal ? 1 : 0;

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
		At<uint8_t>(p.runtime, kRvaWantHistory) = 0;
		At<void*>(p.runtime, kRvaHistory) = nullptr;
	}

	At<UINT>(p.runtime, kRvaDepthInverted) = 0;
	At<uint8_t>(p.runtime, kRvaDepthExplicit) = 1;
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

	At<float>(p.runtime, kRvaLocalTone) =
		std::clamp(p.settings.localToneStrength, 0.0f, 2.0f);
	const float engineStructure =
		std::clamp(p.settings.localStructureStrength * styleDetail, 0.0f, 2.0f);
	const float engineSkin =
		std::clamp(p.settings.skinStructureStrength * styleSkin, 0.0f, 2.0f);
	At<float>(p.runtime, kRvaLocalStructure) = engineStructure;
	At<float>(p.runtime, kRvaSkinStructure) = engineSkin;
	At<UINT>(p.runtime, kRvaToneChannels) = styleChannels;
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
	At<UINT>(p.runtime, kRvaCharMask) = p.settings.useAutoMask ? 1u : 0u;
	At<uint8_t>(p.runtime, kRvaUseDepth) = p.settings.amdUseDepth ? 1 : 0;
	At<uint8_t>(p.runtime, kRvaUseFsrInputs) = p.settings.amdUseFsrInputs ? 1 : 0;
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
	p.Bind(0, p.net.get(), DXGI_FORMAT_R16G16B16A16_FLOAT,
		p.net.get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
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
	packet.colour = p.net.get();
	packet.colourState = kPacketState;
	packet.motion = p.motion.get();
	packet.motionState = kPacketState;
	packet.depth = p.depth.get();
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

	// No transitions here. The engine is handed shader-readable surfaces, exactly as the
	// reference hands it shader-readable surfaces, and it deals with hazards itself.

	if (!CallRecord(p.record, &packet) || At<uint8_t>(p.runtime, kRvaStatusFlag) != 0) {
		p.failed = true;
		Logger::Get().Error("DLSSNR AMD: the engine refused the frame");
		return true;
	}

	if (FAILED(p.list->Close())) {
		p.failed = true;
		Logger::Get().Error("DLSSNR AMD: the command list latched an error");
		return true;
	}
	ID3D12CommandList* lists[] = { p.list.get() };
	p.queue->ExecuteCommandLists(1, lists);
	if (!CallNotify(p.notify, p.queue.get(), 1, lists)) {
		p.failed = true;
		return true;
	}

	// 5 s while the engine is still warming up, 500 ms once it is going. Both are far
	// above the steady-state cost (tens of milliseconds at this resolution) and bound
	// how long a stalled job may hold the render thread.
	const uint64_t budget = p.framesSeen < 3 ? 5000 : 500;
	++p.framesSeen;
	if (!p.WaitForEngine(budget)) {
		// Skip this frame and let the next one try. Latching a failure here would mean
		// one slow frame disables the effect until the profile is reloaded, which is a
		// far worse outcome than a frame that came out unprocessed.
		if (++p.timeouts <= 3) {
			Logger::Get().Warn(fmt::format(
				"DLSSNR AMD: engine did not finish within {} ms; frame passed through "
				"({} so far)", budget, p.timeouts));
		}
		// A frame the engine gave up on leaves its history one step behind the scene.
		p.resetHistory = true;
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

	if (FAILED(p.allocator->Reset()) || FAILED(p.list->Reset(p.allocator.get(), nullptr))) {
		p.failed = true;
		return true;
	}
	// The engine leaves its surface readable, which is the state named below.
	//
	// At a reduced scale a resolve sits here: only the difference the engine made is
	// upscaled, bounded, and weighted by how well the full-resolution original agrees with
	// the low-resolution footprint. The picture is therefore the untouched full-resolution
	// frame plus that edit -- which is what keeps a smaller model from costing detail. At
	// 100% there is nothing to resolve and the engine's own surface is the picture.
	ID3D12Resource* picture = p.net.get();
	if (p.scaled) {
		// The residual for this frame, before anything composites it. With anti-flicker off
		// this only subtracts; with it on, earlier frames are blended in here.
		if (!p.RunTemporal(context)) {
			p.failed = true;
			return true;
		}
		Barrier(p.list.get(), p.resolved.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		p.BindResolve(8, p.full.get(), p.resolved.get(), p.baseline.get(),
			p.historyResidual[p.historyIndex].get());
		// NR Intensity scales the edit the resolve applies -- the same idea as NGX's
		// residual intensity, and what a strength control on this effect should do. The
		// bound is the cap; the multiplier is what the slider moves.
		const uint32_t boundMilli = static_cast<uint32_t>(std::clamp(
			float(p.editBoundMilli) * std::clamp(p.settings.intensity, 0.0f, 2.0f),
			0.0f, 1000.0f));
		const UINT dims[5]{ p.width, p.height, p.netWidth, p.netHeight, boundMilli };
		p.DispatchSized(p.resolve.get(), 8, p.width, p.height, 10, dims, 5);
		Barrier(p.list.get(), p.resolved.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kStateShaderRead);
		picture = p.resolved.get();
	}

	if (p.outputIsFp16) {
		Barrier(p.list.get(), picture, kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(p.list.get(), p.sharedOut12.get(), kCommon,
			D3D12_RESOURCE_STATE_COPY_DEST);
		p.list->CopyResource(p.sharedOut12.get(), picture);
		Barrier(p.list.get(), p.sharedOut12.get(), D3D12_RESOURCE_STATE_COPY_DEST, kCommon);
	} else {
		Barrier(p.list.get(), p.sharedOut12.get(), kCommon,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		p.Bind(6, picture, DXGI_FORMAT_R16G16B16A16_FLOAT,
			p.sharedOut12.get(), p.outputFormat);
		p.list->SetComputeRoot32BitConstants(1, 1, &p.shoulderMilli, 0);
		p.Dispatch(p.srgbInput ? p.convertOutSrgb.get() : p.convertOut.get(), 6);
		Barrier(p.list.get(), p.sharedOut12.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kCommon);
	}
	if (FAILED(p.list->Close())) {
		p.failed = true;
		return true;
	}
	p.queue->ExecuteCommandLists(1, lists);

	const uint64_t signal2 = ++p.fenceValue;
	p.queue->Signal(p.fence.get(), signal2);
	if (p.fence->GetCompletedValue() < signal2) {
		p.fence->SetEventOnCompletion(signal2, p.fenceEvent.get());
		if (WaitForSingleObject(p.fenceEvent.get(), 1000) != WAIT_OBJECT_0) {
			p.failed = true;
			return true;
		}
	}
	// The frame's last signal: the guidance textures may be reused by the producer once this
	// value is reached, so the interop is told it before the next frame can Update.
	if (p.guidanceInterop) {
		p.guidanceInterop->MarkSubmitted(signal2);
	}

	p.context11->CopyResource(context.output, p.sharedOut11.get());

	// TEMPORARY DIAGNOSTIC -- remove once the brightness wobble is pinned down.
	//
	// Twelve consecutive frames' means instead of one sample. A single reading cannot tell a
	// steady picture from one that is breathing; a run of them can, and the three points
	// separate the frame arriving from the engine's own output from what finally leaves, so
	// whichever of the three is swinging is where to look. Readbacks stall the queue, which
	// is why this is a short window and not a permanent per-frame reading.
	if (p.probeSamples < 12 && p.framesSeen >= 10) {
		const int n = p.probeSamples++;
		uint64_t badIn = 0, badNet = 0, badOut = 0;
		float deltaIn = -1.0f, deltaOut = -1.0f;
		// Both ends are measured frame to frame. The input's change is what the game did;
		// the picture's change is what flicker is when the scene is still.
		const float inMean = p.Measure(p.sharedIn12.get(), p.inputFormat,
			D3D12_RESOURCE_STATE_COMMON, &badIn, &deltaIn);
		const float netMean = p.Measure(p.net.get(), DXGI_FORMAT_R16G16B16A16_FLOAT,
			kStateShaderRead, &badNet);
		const float outMean = p.Measure(p.sharedOut12.get(), p.outputFormat,
			D3D12_RESOURCE_STATE_COMMON, &badOut, &deltaOut);
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD series {:2d}: in {:.4f}(bad {}) net {:.4f}(bad {}) "
			"out {:.4f}(bad {}) | frame-to-frame: in {:.5f} out {:.5f}",
			n, inMean, badIn, netMean, badNet, outMean, badOut, deltaIn, deltaOut));
	}
	return true;
}

bool DlssnrAmdBackend::Drain() noexcept {
	return true;
}

}
