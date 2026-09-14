#include "pch.h"
#include "DlssnrAmdBackend.h"
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

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	dst[id.xy] = src.Load(int3(id.xy, 0));
}
)";

constexpr char kConvertOutShader[] = R"(
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	dst[id.xy] = src.Load(int3(id.xy, 0));
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
Texture2D<float4> edited : register(t2);
RWTexture2D<float4> dst : register(u0);
cbuffer Extent : register(b0) { uint w; uint h; uint lowW; uint lowH; };
float3 delta(int2 p) {
	p = clamp(p, 0, int2(lowW - 1, lowH - 1));
	return edited.Load(int3(p, 0)).rgb - baseline.Load(int3(p, 0)).rgb;
}
[numthreads(8, 8, 1)]
void main(uint3 p : SV_DispatchThreadID)
{
	if (p.x >= w || p.y >= h) return;
	float2 q = (float2(p.xy) + .5) * float2(lowW, lowH) / float2(w, h) - .5;
	int2 a = int2(floor(q));
	float2 t = frac(q);
	float3 d = lerp(lerp(delta(a), delta(a + int2(1, 0)), t.x),
		lerp(delta(a + int2(0, 1)), delta(a + 1), t.x), t.y);
	float4 c = src.Load(int3(p.xy, 0));
	// A reduced neural pixel mixes surfaces and small emitters. Suppress its edit where the
	// original pixel disagrees with that footprint, rather than spreading the edit blindly
	// across high-contrast edges.
	int2 hi = int2(lowW - 1, lowH - 1);
	float3 b = lerp(lerp(baseline.Load(int3(clamp(a, 0, hi), 0)).rgb,
			baseline.Load(int3(clamp(a + int2(1, 0), 0, hi), 0)).rgb, t.x),
		lerp(baseline.Load(int3(clamp(a + int2(0, 1), 0, hi), 0)).rgb,
			baseline.Load(int3(clamp(a + 1, 0, hi), 0)).rgb, t.x), t.y);
	float3 magnitude = max(max(abs(c.rgb), abs(b)), 1e-5);
	float mismatch = max(abs(c.r - b.r) / magnitude.r,
		max(abs(c.g - b.g) / magnitude.g, abs(c.b - b.b) / magnitude.b));
	float confidence = 1 - smoothstep(.15, .75, mismatch);
	// Keep extreme low-resolution edits bounded relative to the current footprint.
	float3 limit = .5 * max(abs(b), abs(c.rgb));
	d = clamp(d, -limit, limit) * confidence;
	dst[p.xy] = float4(clamp(c.rgb + d, 0, 65504), c.a);
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
	winrt::com_ptr<ID3D12PipelineState> convertOut;
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
	void Dispatch(ID3D12PipelineState* pso, uint32_t slot) noexcept;
	void DispatchSized(ID3D12PipelineState* pso, uint32_t slot, uint32_t dw, uint32_t dh,
		bool setResidualTable, const UINT* dims4 = nullptr) noexcept;
	bool WaitForEngine(uint64_t deadlineMs) noexcept;
	void DestroySized() noexcept;
	float Measure(ID3D12Resource* res, DXGI_FORMAT format,
		D3D12_RESOURCE_STATES before, uint64_t* nonFinite = nullptr) noexcept;
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
	D3D12_DESCRIPTOR_RANGE ranges[2]{};
	ranges[0] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0 };
	ranges[1] = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 1 };

	D3D12_ROOT_PARAMETER params[3]{};
	params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	params[0].DescriptorTable = { 2, ranges };
	params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	params[1].Constants = { 0, 0, 24 };
	params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	D3D12_DESCRIPTOR_RANGE residualRange{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1, 0, 0 };
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
		winrt::com_ptr<ID3D12PipelineState>& out) -> bool {
		winrt::com_ptr<ID3DBlob> cs, csError;
		if (FAILED(D3DCompile(source, length, name, nullptr, nullptr, "main",
			"cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, cs.put(), csError.put()))) {
			Logger::Get().Error(fmt::format("DLSSNR AMD: {} shader failed: {}", name,
				csError ? static_cast<const char*>(csError->GetBufferPointer()) : "?"));
			return false;
		}
		D3D12_COMPUTE_PIPELINE_STATE_DESC ps{};
		ps.pRootSignature = root.get();
		ps.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
		return SUCCEEDED(device12->CreateComputePipelineState(&ps,
			IID_PPV_ARGS(out.put())));
	};

	return compile(kConvertInShader, sizeof(kConvertInShader) - 1, "convert in",
			convertIn) &&
		compile(kConvertOutShader, sizeof(kConvertOutShader) - 1, "convert out",
			convertOut) &&
		compile(kDownsampleShader, sizeof(kDownsampleShader) - 1, "downsample",
			downsample) &&
		compile(kResolveShader, sizeof(kResolveShader) - 1, "resolve", resolve);
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
	// The short side is held at 540 pixels, and that floor is measured rather than assumed.
	// At 480x270 -- 25% of a 1080p capture -- two things go wrong at once. The engine's
	// output falls to about a sixteenth of its input (probe: `frame in 0.4905, engine out
	// 0.0312`, against 0.5873/0.5908 at full size), so the filter is not filtering. And its
	// auto-exposure starts hunting: on a scene whose input mean sat between 0.490 and 0.502
	// the exposure the engine chose swung between 0.645 and 0.925, a fifth of its own value,
	// frame to frame. That is what reads as an old projector lamp breathing.
	//
	// 540 is the convention for the smallest frame this class of model is built for, and it
	// has the useful property of landing both common captures inside the range that works:
	// a 1080p capture stops at 50%, a 4K capture can reach 25% and still be 960x540.
	constexpr uint32_t kMinimumNetworkShortSide = 540;
	const uint32_t scaledW = uint32_t(float(w) * modelScale + 0.5f);
	const uint32_t scaledH = uint32_t(float(h) * modelScale + 0.5f);
	// The scale that would put the short side exactly on the floor, in ten-thousandths, so
	// the two dimensions keep their aspect. The request is raised to meet it rather than
	// being rejected, so asking for too small a frame still gives the smallest usable one.
	const uint32_t shortSide = std::min(w, h);
	const uint32_t floorScale = shortSide <= kMinimumNetworkShortSide ? 10000u
		: uint32_t(uint64_t(kMinimumNetworkShortSide) * 10000u / shortSide);
	netWidth = std::max(std::clamp(scaledW, 32u, w),
		uint32_t(uint64_t(w) * floorScale / 10000u));
	netHeight = std::max(std::clamp(scaledH, 32u, h),
		uint32_t(uint64_t(h) * floorScale / 10000u));
	scaled = netWidth != w || netHeight != h;

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
	// A new extent means the history describes the wrong geometry.
	resetHistory = true;
	motion = nullptr;
	depth = nullptr;
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
	uint32_t dw, uint32_t dh, bool setResidualTable, const UINT* dims4
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
	if (setResidualTable) {
		// The resolve reads its two extra sources from t1/t2, which live in the table two
		// descriptors along.
		D3D12_GPU_DESCRIPTOR_HANDLE residual = table;
		residual.ptr += 2 * descriptorStride;
		list->SetComputeRootDescriptorTable(2, residual);
	}
	list->SetPipelineState(pso);
	// The downsample and the resolve are the only stages with an extent to pass, and they
	// take it in the same four-slot shape the reference uses: destination width and height
	// followed by the source's.
	if (dims4) {
		list->SetComputeRoot32BitConstants(1, 4, dims4, 0);
	}
	list->Dispatch((dw + 7) / 8, (dh + 7) / 8, 1);
}

void DlssnrAmdBackend::Impl::BindResolve(uint32_t slot, ID3D12Resource* srv,
	ID3D12Resource* uav, ID3D12Resource* baselineTexture, ID3D12Resource* edited
) noexcept {
	// t0/u0 through the usual pair, then t1 and t2 sitting immediately after it so the
	// resolve's second root table finds them two descriptors along.
	Bind(slot, srv, DXGI_FORMAT_R16G16B16A16_FLOAT, uav, DXGI_FORMAT_R16G16B16A16_FLOAT);

	D3D12_SHADER_RESOURCE_VIEW_DESC s{};
	s.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
	s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	s.Texture2D.MipLevels = 1;
	device12->CreateShaderResourceView(baselineTexture, &s, Cpu(slot + 2));
	device12->CreateShaderResourceView(edited, &s, Cpu(slot + 3));
}

bool DlssnrAmdBackend::Impl::InitEngine(const std::filesystem::path& weightsPath) noexcept {
	At<ID3D12Device*>(runtime, kRvaDevice) = device12.get();
	device12.get()->AddRef();
	At<ID3D12CommandQueue*>(runtime, kRvaQueue) = queue.get();
	queue.get()->AddRef();
	At<int>(runtime, kRvaHipDevice) = hipDevice;
	At<uint8_t>(runtime, kRvaInlineMode) = 1;
	At<uint8_t>(runtime, kRvaInterop) = 1;
	At<uint8_t>(runtime, kRvaEnabled) = 1;
	At<uint8_t>(runtime, kRvaUseFsrInputs) = 1;
	At<uint8_t>(runtime, kRvaUseDepth) = 0;

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
	const DLSSNRSettings& settings
) noexcept {
	auto& p = *_impl;
	// Recorded, not yet acted on. The engine's fields are still written from this
	// backend's own constants because mapping Magpie's parameter names onto them one-to-one
	// does not work -- see the note in Draw. Printing them means the next attempt starts
	// from what Magpie actually passes rather than from what the UI claims its defaults are.
	Logger::Get().Info(fmt::format(
		"DLSSNR AMD parameters: tone={:.3f} structure={:.3f} skin={:.3f} autoMask={} "
		"intensity={:.3f} style={} resolution={}%",
		settings.localToneStrength, settings.localStructureStrength,
		settings.skinStructureStrength, settings.useAutoMask ? "on" : "off",
		settings.intensity, settings.style,
		settings.enableInputResolutionScaling ? int(settings.inputResolutionPercent) : 100));
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
		"(requested {:.2f}, applied {:.2f})", p.hipDevice, inputDesc.Width, inputDesc.Height,
		(uint32_t)inputDesc.Format, (uint32_t)outputDesc.Format,
		p.netWidth, p.netHeight, p.modelScale, appliedScale));
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
	uint64_t* nonFinite
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
	if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
		const uint16_t* p16 = static_cast<const uint16_t*>(mapped);
		for (uint64_t i = 0; i < total / 2; ++i) {
			const float v = HalfToFloat(p16[i]);
			if (std::isfinite(v)) { sum += v; ++count; } else { ++bad; }
		}
	} else {
		const uint8_t* p8 = static_cast<const uint8_t*>(mapped);
		for (uint64_t i = 0; i < total; ++i) { sum += p8[i] / 255.0; ++count; }
	}
	probe->Unmap(0, nullptr);
	if (nonFinite) {
		*nonFinite = bad;
	}
	return count ? float(sum / double(count)) : -1.0f;
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
		p.Dispatch(p.convertIn.get(), 2);
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
	if (p.scaled) {
		Barrier(p.list.get(), p.net.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		p.Bind(4, p.full.get(), DXGI_FORMAT_R16G16B16A16_FLOAT, p.net.get(),
			DXGI_FORMAT_R16G16B16A16_FLOAT);
		const UINT dims[4]{ p.netWidth, p.netHeight, p.width, p.height };
		p.DispatchSized(p.downsample.get(), 4, p.netWidth, p.netHeight, false, dims);
		Barrier(p.list.get(), p.net.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			kStateShaderRead);

		Barrier(p.list.get(), p.net.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_SOURCE);
		Barrier(p.list.get(), p.baseline.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_COPY_DEST);
		p.list->CopyResource(p.baseline.get(), p.net.get());
		Barrier(p.list.get(), p.baseline.get(), D3D12_RESOURCE_STATE_COPY_DEST,
			kStateShaderRead);
		Barrier(p.list.get(), p.net.get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
			kStateShaderRead);
	} else {
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
		p.Dispatch(p.convertOut.get(), 6);
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
	At<uint8_t>(p.runtime, kRvaPerPassFlag) = 1;

	// History is only meaningful while consecutive frames agree. The engine's own job
	// counter cannot be the trigger, for the reason the reference gives: recreating staging
	// restarts it, so job 1 can follow job 1 and equality says nothing.
	const uint64_t now = GetTickCount64();
	const bool gap = p.lastSubmitTick != 0 && now - p.lastSubmitTick > 250;
	p.lastSubmitTick = now;
	if (p.resetHistory || gap) {
		p.resetHistory = false;
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
	At<float>(p.runtime, kRvaLocalTone) = 0.0f;
	At<float>(p.runtime, kRvaLocalStructure) = 1.0f;
	At<float>(p.runtime, kRvaSkinStructure) = 1.0f;
	At<UINT>(p.runtime, kRvaToneChannels) = 0;
	At<UINT>(p.runtime, kRvaCharMask) = 1;
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
	packet.scaleX = 1.0f;
	packet.scaleY = 1.0f;

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
		Barrier(p.list.get(), p.resolved.get(), kStateShaderRead,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		p.BindResolve(8, p.full.get(), p.resolved.get(), p.baseline.get(), p.net.get());
		const UINT dims[4]{ p.width, p.height, p.netWidth, p.netHeight };
		p.DispatchSized(p.resolve.get(), 8, p.width, p.height, true, dims);
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
		p.Dispatch(p.convertOut.get(), 6);
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
		const float inMean = p.Measure(p.sharedIn12.get(), p.inputFormat,
			D3D12_RESOURCE_STATE_COMMON, &badIn);
		const float netMean = p.Measure(p.net.get(), DXGI_FORMAT_R16G16B16A16_FLOAT,
			kStateShaderRead, &badNet);
		const float outMean = p.Measure(p.sharedOut12.get(), p.outputFormat,
			D3D12_RESOURCE_STATE_COMMON, &badOut);
		Logger::Get().Info(fmt::format(
			"DLSSNR AMD series {:2d}: in {:.4f}(bad {}) net {:.4f}(bad {}) out {:.4f}(bad {})",
			n, inMean, badIn, netMean, badNet, outMean, badOut));
	}
	return true;
}

bool DlssnrAmdBackend::Drain() noexcept {
	return true;
}

}
