#pragma once
#include <chrono>

#include "NativeEffectBackend.h"
#include "GroupBEffectProtocol.h"

namespace Magpie {

class DeviceResources;

// Experimental FSR 3.1.5 / FSR 4.1.1 upscaler running on D3D12 through
// resources shared with Magpie's D3D11 renderer. Frame generation is omitted.
class FSR3Upscaler final : public NativeEffectBackend {
public:
	struct Impl;

	FSR3Upscaler();
	FSR3Upscaler(const FSR3Upscaler&) = delete;
	FSR3Upscaler& operator=(const FSR3Upscaler&) = delete;
	~FSR3Upscaler() override;

	bool Initialize(DeviceResources& resources, ID3D11Texture2D* input,
		ID3D11Texture2D* output, MotionVectorRequest motionRequest = {}, bool useFsr4 = false) noexcept;
	bool Resize(DeviceResources& resources, ID3D11Texture2D* input,
		ID3D11Texture2D* output) noexcept override;
	bool Draw(const NativeEffectDrawContext& context) noexcept override;
	void SetFsrHdrProtocol(const FsrHdrProtocol& protocol) noexcept {
		_hdrProtocol = protocol;
	}
	// The sub-pixel offset this frame's input was sampled at, in the input's own pixels. The
	// upscaler accumulates across frames, and a sequence of frames that were all sampled at
	// the same place carries no new information to accumulate -- which reads as softness.
	void SetJitter(float x, float y) noexcept {
		_jitterX = x;
		_jitterY = y;
	}
	FrameGuidanceRequirements GetFrameGuidanceRequirements() const noexcept override {
		FrameGuidanceRequirements result{ .zero = true };
		result.Add(_motionRequest);
		return result;
	}

	EffectParameterRestartReason GetParameterRestartReason(std::string_view) const noexcept override {
		return EffectParameterRestartReason::FrameGuidance;
	}

private:
	MotionVectorRequest _motionRequest{};
	[[maybe_unused]] std::unique_ptr<Impl> _impl;
	[[maybe_unused]] bool _useFsr4 = false;
	FsrHdrProtocol _hdrProtocol{};
	float _jitterX = 0.0f;
	float _jitterY = 0.0f;
	// When the previous dispatch happened, and the interval that came out of it.
	//
	// FSR is told how long the frame took, in milliseconds, and that number is the weight its
	// temporal accumulation is built on. It stood at 16.6667 -- the value for 60 fps -- while
	// this chain was producing a frame every hundred milliseconds or so, so the history was
	// being treated as seven times fresher than it was.
	std::chrono::steady_clock::time_point _lastDispatch{};
	float _frameDeltaMs = 16.6667f;
};

}
