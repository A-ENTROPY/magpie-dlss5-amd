// Experimental NVIDIA DLSS neural same-resolution filter. The native D3D12
// backend replaces this pass and uses shared optical flow with zero depth.
//
// Network options 0, 4 and 5 run on the Vulkan runtime. Two things about it are
// expected rather than broken:
//   * The picture goes black for a moment when scaling starts. The runtime builds its
//     network on first use and refuses frames until it is ready; those frames are
//     skipped, so the screen shows the desktop or a frozen frame rather than the game.
//     The first run also compiles 109 pipelines, which takes about a minute, and it is
//     cached afterwards.
//   * The network runs as a pure spatial filter: no temporal accumulation is applied
//     here, because on this path the history is fed estimated flow rather than the
//     game's motion vectors and carries more error than it removes.
//!MAGPIE EFFECT
//!VERSION 4
//!SORT_NAME DLSSNR AI Filter (Experimental)

//!PARAMETER
//!GROUP Detail Control
//!LABEL Network Size\n(needs a restart; 1440/2160 Vulkan-only, slow)
//!DEFAULT 0
//!OPTION 0 Auto (the runtime picks)
//!OPTION 360 640x360 - fastest
//!OPTION 576 1024x576
//!OPTION 720 1280x720
//!OPTION 900 1600x900
//!OPTION 1080 1920x1080 - best quality
//!OPTION 1440 2560x1440 - Vulkan only, very slow
//!OPTION 2160 3840x2160 - Vulkan only, capture size, ~13 fps
int networkTier;

//!PARAMETER
//!GROUP Detail Control
//!LABEL Adjust Input Resolution\n(Reduces DLSSNR Quality)
//!DEFAULT 0
//!MIN 0
//!MAX 1
//!STEP 1
int enableInputResolutionScaling;

//!PARAMETER
//!GROUP Detail Control
//!LABEL Input Resolution (%)
//!DEFAULT 100
//!MIN 25
//!MAX 100
//!STEP 1
int inputResolutionPercent;

//!PARAMETER
//!GROUP Detail Control
//!LABEL Residual Multiplier
//!DEFAULT 1
//!MIN 1
//!MAX 2
//!STEP 0.05
float residualMultiplier;

//!PARAMETER
//!GROUP Detail Control
//!LABEL Residual Saturation Multiplier
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float residualSaturation;

//!PARAMETER
//!GROUP Detail Control
//!LABEL Residual Lightness Multiplier
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float residualLightness;

//!PARAMETER
//!GROUP Detail Control
//!LABEL Shadow / Structure Control
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float shadowStructureMultiplier;

//!PARAMETER
//!GROUP Detail Control
//!LABEL Reflection / Glow Control
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float reflectionGlowMultiplier;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL NR Style\n(0 Default, 1 Natural, 2 Cinematic)
//!DEFAULT 0
//!MIN 0
//!MAX 2
//!STEP 1
int style;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL NR Intensity
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float intensity;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL Local Tone Strength
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float localToneStrength;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL Local Structure Strength
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float localStructureStrength;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL Skin Structure Strength
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float skinStructureStrength;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL Automatic Mask
//!DEFAULT 1
//!MIN 0
//!MAX 1
//!STEP 1
int useAutoMask;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL NR UI Correction
//!DEFAULT 0
//!MIN 0
//!MAX 1
//!STEP 1
int uiCorrection;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL Temporal History
//!DEFAULT 1
//!MIN 0
//!MAX 1
//!STEP 1
int amdTemporal;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL Tone Channels
//!DEFAULT 0
//!MIN 0
//!MAX 2
//!STEP 1
int amdToneChannels;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL Use Depth Guide
//!DEFAULT 0
//!MIN 0
//!MAX 1
//!STEP 1
int amdUseDepth;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL Use Host Inputs
//!DEFAULT 1
//!MIN 0
//!MAX 1
//!STEP 1
int amdUseFsrInputs;

//!PARAMETER
//!GROUP DLSSNR · Pass 1
//!LABEL NR Reconstruction
//!DEFAULT 1
//!OPTION 0 Resolve (residual)
//!OPTION 1 FSR 4.1.1 upscale
int amdReconstruct;

//!PARAMETER
//!GROUP DLSSNR · Model
//!LABEL Network\n(runtime + block count; changing it rebuilds the session)
//!DEFAULT 0
//!OPTION 0 Small 31-block - Vulkan (fastest)
//!OPTION 1 Small 31-block - pak
//!OPTION 2 Large 68-block - pak
//!OPTION 3 Large 71-block - vendored (~13 fps at 4K)
//!OPTION 4 Large 71-block - Vulkan
//!OPTION 5 Large 71-block - Vulkan, int4 weights
int networkModel;

//!PARAMETER
//!GROUP DLSSNR · Layers
//!LABEL Layer Count\n(stacks the network; needs a restart)
//!DEFAULT 1
//!OPTION 1 Single layer
//!OPTION 2 Two layers
//!OPTION 3 Three layers
int multiPass;

//!PARAMETER
//!GROUP DLSSNR · Layer 2
//!LABEL Style\n(0 Def / 1 Nat / 2 Cine)
//!DEFAULT 0
//!OPTION 0 Standard
//!OPTION 1 Natural
//!OPTION 2 Cinematic
int pass2_style;

//!PARAMETER
//!GROUP DLSSNR · Layer 2
//!LABEL Intensity
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float pass2_intensity;

//!PARAMETER
//!GROUP DLSSNR · Layer 2
//!LABEL Local Structure
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float pass2_localStructureStrength;

//!PARAMETER
//!GROUP DLSSNR · Layer 2
//!LABEL Skin Structure
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float pass2_skinStructureStrength;

//!PARAMETER
//!GROUP DLSSNR · Layer 2
//!LABEL Automatic Mask
//!DEFAULT 1
//!OPTION 0 Off
//!OPTION 1 On
int pass2_useAutoMask;

//!PARAMETER
//!GROUP DLSSNR · Layer 3
//!LABEL Style\n(0 Def / 1 Nat / 2 Cine)
//!DEFAULT 0
//!OPTION 0 Standard
//!OPTION 1 Natural
//!OPTION 2 Cinematic
int pass3_style;

//!PARAMETER
//!GROUP DLSSNR · Layer 3
//!LABEL Intensity
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float pass3_intensity;

//!PARAMETER
//!GROUP DLSSNR · Layer 3
//!LABEL Local Structure
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float pass3_localStructureStrength;

//!PARAMETER
//!GROUP DLSSNR · Layer 3
//!LABEL Skin Structure
//!DEFAULT 1
//!MIN 0
//!MAX 2
//!STEP 0.05
float pass3_skinStructureStrength;

//!PARAMETER
//!GROUP DLSSNR · Layer 3
//!LABEL Automatic Mask
//!DEFAULT 1
//!OPTION 0 Off
//!OPTION 1 On
int pass3_useAutoMask;

//!PARAMETER
//!GROUP DLSSNR · Optical Flow
//!LABEL Optical Flow Method\n(0 none / 1 AMD / 2 NVIDIA; unused on the Vulkan runtime)
//!DEFAULT 0
//!OPTION 0 None - no motion vectors, zero-MV route
//!OPTION 1 AMD Optical Flow - uses AMD's hardware flow
//!OPTION 2 NVIDIA Optical Flow - uses NVIDIA's flow
int opticalFlowMethod;

//!PARAMETER
//!GROUP DLSSNR · Optical Flow
//!LABEL AMD OF Quality\n(only when method = 1)
//!DEFAULT 1
//!OPTION 0 Performance
//!OPTION 1 Quality
int amdOpticalFlowMode;

//!PARAMETER
//!GROUP DLSSNR · Optical Flow
//!LABEL NVOF Quality\n(only when method = 2)
//!DEFAULT 2
//!MIN 1
//!MAX 5
//!STEP 1
int nvidiaOpticalFlowQuality;




//!PARAMETER
//!GROUP 内核级抗闪烁
//!LABEL 时域稳定
//!DEFAULT 1
//!OPTION 0 关闭
//!OPTION 1 开启（按邻域约束承载的编辑）
int temporalClamp;

//!PARAMETER
//!GROUP 内核级抗闪烁
//!LABEL 720 档增强
//!DEFAULT 1
//!OPTION 0 关闭
//!OPTION 1 开启（小网络编辑更粗，需要更强约束）
int temporalClampMid;

//!PARAMETER
//!GROUP 内核级抗闪烁
//!LABEL 运动自适应
//!DEFAULT 1
//!OPTION 0 关闭
//!OPTION 1 开启（运动时放松约束，静止时收紧）
int temporalClampMidMotion;

//!PARAMETER
//!GROUP 内核级抗闪烁
//!LABEL 遮挡处理
//!DEFAULT 1
//!OPTION 0 关闭
//!OPTION 1 开启（表面重新入画时丢弃承载编辑）
int temporalDisocc;

//!PARAMETER
//!GROUP 内核级抗闪烁
//!LABEL 输出平滑
//!DEFAULT 0
//!OPTION 0 关闭
//!OPTION 1 开启（更稳，但可能略软）
int passSmooth;

//!PARAMETER
//!GROUP 内核级抗闪烁
//!LABEL 静止帧保持
//!DEFAULT 0
//!OPTION 0 关闭
//!OPTION 1 开启（画面静止时保持上一帧编辑）
int passStill;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//!SAMPLER
//!FILTER LINEAR
SamplerState sam;

//!PASS 1
//!STYLE PS
//!IN INPUT
//!OUT OUTPUT

MF4 Pass1(float2 pos) {
	return INPUT.SampleLevel(sam, pos, 0);
}
