#pragma once
#include "FrameGuidanceTypes.h"
#include "ScalingOptions.h"
#include "HdrEffectBoundary.h"
#include <utility>

namespace Magpie {

class DeviceResources;

struct NativeEffectDrawContext {
	ID3D11Texture2D* input = nullptr;
	ID3D11Texture2D* output = nullptr;
	HdrFrameMetadata inputMetadata{};
	HdrFrameMetadata outputMetadata{};
	FrameGuidanceFrameId frameId = 0;
	// Changes whenever an earlier effect in the chain changes its output for
	// the same captured frame. Native effects that cache duplicate frames must
	// include this value in their cache key.
	uint64_t inputRevision = 0;
	const FrameGuidanceView& frameGuidance;
	const FrameGuidanceView& zeroFrameGuidance;
};

// Common lifetime and rendering contract for native SDK-backed effects.
// Creation parameters remain in NativeEffectBackendFactory so Renderer does
// not need one parallel container and one name-dispatch branch per SDK.
class NativeEffectBackend {
public:
	virtual ~NativeEffectBackend() = default;

	virtual void SetHdrBoundary(HdrEffectBoundaryContext context) noexcept { _hdrBoundary = std::move(context); }
	const HdrEffectBoundaryContext& GetHdrBoundary() const noexcept { return _hdrBoundary; }

	virtual FrameGuidanceRequirements GetFrameGuidanceRequirements() const noexcept {
		return {};
	}
	virtual bool Drain() noexcept { return true; }

	virtual EffectParameterApplyMode GetParameterApplyMode(
		std::string_view /*parameterName*/
	) const noexcept {
		return EffectParameterApplyMode::RestartRequired;
	}

	virtual EffectParameterRestartReason GetParameterRestartReason(
		std::string_view /*parameterName*/
	) const noexcept {
		return EffectParameterRestartReason::NativeBackend;
	}

	// Called on the backend thread with a complete candidate option. Implementations
	// must not recreate resources or partially commit settings when validation fails.
	virtual bool ApplyLiveParameters(
		const EffectOption& /*option*/,
		std::span<const std::string> /*parameterNames*/
	) noexcept {
		return false;
	}

	// Backend-thread transaction, serialized with Draw/Resize/destruction.
	// A model-backed implementation may load a candidate private model here;
	// renderer-owned textures and the active model survive a failed load.
	virtual bool ApplyParameters(const EffectOption& option,
		std::span<const std::string> names) noexcept {
		return ApplyLiveParameters(option, names);
	}

	virtual bool Resize(
		DeviceResources& resources,
		ID3D11Texture2D* input,
		ID3D11Texture2D* output
	) noexcept = 0;

	// Whether this backend can still produce a result, for a chain that has to decide whether
	// to composite the frame at all. Both implementations latch: false means a failure that
	// does not clear, not a frame that was merely dropped.
	//
	// Every site that sets it is unrecoverable by construction -- a removed device, a command
	// list that will not reset, an engine that refuses to record, a fence that timed out. The
	// surfaces are left holding whatever they held, so retrying would composite a stale or
	// torn frame rather than recover one. A recoverable miss returns false from Draw alone and
	// leaves this true.
	//
	// Default true: a backend with no notion of failure is healthy by construction.
	virtual bool IsHealthy() const noexcept { return true; }

	virtual bool Draw(const NativeEffectDrawContext& context) noexcept = 0;

protected:
	HdrEffectBoundaryContext _hdrBoundary{};
};

}
