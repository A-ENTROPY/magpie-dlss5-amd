#pragma once
#include "NativeEffectBackend.h"
#include "DLSSNRFilter.h"

namespace Magpie {

class DeviceResources;

// DLSS Neural Rendering on an AMD GPU, through the DLSS-NR-on-AMD runtime.
//
// Magpie's existing DLSSNR effect drives NVIDIA's NGX feature 18, which only exists
// on RTX hardware. This backend drives the separate AMD port of the same network
// instead: a DLL that carries HIP kernels for the AMD architectures and is called
// through fixed offsets inside its own image.
//
// It is deliberately independent of NgxD3D12Core. That class is compiled out entirely
// when the NGX SDK is absent and initialises NGX when it is present; neither is wanted
// here. The D3D12 device this backend needs is created directly on the same adapter as
// Magpie's D3D11 device, and the frame crosses between the two through a shared
// texture, which is the same mechanism DLSSNRFilter already uses.
class DlssnrAmdBackend final : public NativeEffectBackend {
public:
	struct Impl;

	DlssnrAmdBackend();
	DlssnrAmdBackend(const DlssnrAmdBackend&) = delete;
	DlssnrAmdBackend& operator=(const DlssnrAmdBackend&) = delete;
	~DlssnrAmdBackend() override;

	// True when the runtime and its weights are both present in the directory the
	// effect will load them from. The factory asks before constructing anything, so a
	// machine without them reports the effect as unavailable instead of failing late.
	static bool IsAvailable() noexcept;

	// The renderer only produces frame guidance -- real and zero -- for backends that
	// declare a need for it. Without this the guidance service never runs optical flow for
	// us, which is why the first motion attempt saw empty views for both.
	FrameGuidanceRequirements GetFrameGuidanceRequirements() const noexcept override;

	bool Initialize(
		DeviceResources& resources,
		ID3D11Texture2D* input,
		ID3D11Texture2D* output,
		const DLSSNRSettings& settings,
		// The effect's anti-flicker request. Modes above 1 reproject history with optical
		// flow, which this backend has no source for, so they are served as mode 1.
		int antiFlickerMode
	) noexcept;

	bool Resize(
		DeviceResources& resources,
		ID3D11Texture2D* input,
		ID3D11Texture2D* output
	) noexcept override;

	bool Draw(const NativeEffectDrawContext& context) noexcept override;
	bool Drain() noexcept override;

private:
	std::unique_ptr<Impl> _impl;
};

}
