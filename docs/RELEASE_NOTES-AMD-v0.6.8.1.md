# 大力喜鹊 AMD RDNA3 v0.6.8.1

大力喜鹊 (Magpie experimental 0.6.8) 的 **AMD Radeon RDNA3 适配版**，在 RX 7900 XTX（gfx1100）上运行 DLSS Neural Rendering（DLSSNR），不依赖 NVIDIA NGX。实测平台：RX 7900 XTX + 赛博朋克 2077 / GTA V Enhanced。

本版是 0.6.8 的补丁：**换用新版 DLSS-NR-on-AMD 0.3.0 运行时**、给 DLSSNR 加了**链内 FSR3 重建**，并把 **XeSS 帧生成的倍率扩到 6×**。

---

## 一、两个包，按需取用

| | 包含 | 适合谁 |
|---|---|---|
| **`magpie-dlss5-amd-rdna3-full.zip`** | **开箱即用**：本适配的 Magpie + 运行时 + 权重 + FSR3 上采样 DLL + XeSS 帧生成 DLL + 全部许可证 | 想解压就能用 |
| **`magpie-dlss5-amd-rdna3.zip`** | 只有本适配的构建产物 | 想自己准备第三方二进制 |

**full 包不需要你再放任何文件**，里面已经有：

    DLSSNR 运行时（0.3.0）        dlssnr_amd_pass1.dll
    网络权重                      dlssnr_on_amd_weights.bin
    运行时设置                    dlssnr_on_amd.ini
    FSR3 重建路线                 amd_fidelityfx_upscaler_dx12.dll
                                  amd_fidelityfx_loader_dx12.dll
    XeSS 帧生成（含多帧）          libxess_fg.dll（1.3.1.78）+ libxell.dll
    XeSS 超分效果                 libxess.dll
    许可证与第三方声明            AMD-FSR-SDK-THIRD-PARTY.md
                                  INTEL-XESS-LICENSE.txt
                                  INTEL-XESS-THIRD-PARTY.txt

注意 `libxell.dll` 是 `libxess_fg.dll` 自己的依赖——只放后者帧生成起不来，所以两个都要。

**干净包**要你自己准备（放在 `Magpie.exe` 旁边）：

- `dlssnr_amd_pass1.dll` — AMD 侧运行时。**必须是 0.3.0**（见下）。
- `dlssnr_on_amd_weights.bin` — 网络权重。
- `dlssnr_on_amd.ini` — 随便建一个文本文件即可（默认值够用）。
- 想用 FSR3 重建路线：`amd_fidelityfx_upscaler_dx12.dll` 与 `amd_fidelityfx_loader_dx12.dll`。
- 想用 XeSS 帧生成：`libxess_fg.dll`（**必须 1.3.1.78**）+ `libxell.dll`。
- 想用 XeSS 超分效果：`libxess.dll`。

**来源说明**：运行时与权重来自第三方的 **OptiScaler AMD pre-SR 包**（该包把神经渲染接进 OptiScaler，超分交给 FFX/FSR），其中运行时本体是 **DLSS-NR on AMD 0.3.0**，原作者 danielblnc。这些文件属于 NVIDIA 派生材料，仅随 full 包提供、不在此授权进一步转发；走合规渠道请用干净包自备。

---

## 二、本版更新

### 1. DLSSNR 适配新版运行时 DLSS-NR-on-AMD 0.3.0

旧版是 2.21。**0.3.0 和 2.21 的内存布局没有一个地址是相同的**，所以本版是这样处理的：

- 启动时**校验运行时的 SHA-256**（0.3.0 = `8321cae7…`）。不匹配就明确报错、拒绝启用，而不是照旧偏移乱跳——**旧运行时现在会被拒**，这是有意行为，不是崩溃。
- 0.3.0 要求每帧多传 16 字节信息（渲染分辨率与颜色的子像素相位）。少传不是"差一点"：引擎会读越界，画面直接全黑。补齐这两个字段后，采样抹掉的细节才能被重建出来。

如果你之前在 2.21 上用得好，升级 Magpie 的同时**必须一起更新运行时**，否则 DLSSNR 不会工作。

### 2. 新增 NR Reconstruction：把 DLSSNR 接进 FSR3 重建

效果面板里多了一个 **NR Reconstruction** 选项：

- **`0 Resolve (residual)`**（默认）——引擎编辑后的帧按残差就地 resolve，就是原来的行为。
- **`1 FSR3 upscale`**——把引擎的输出交给本项目内已有的 FSR3 上采样器，重建到显示分辨率，并接上引擎的 HDR 协议和同尺寸的运动引导。

**怎么选**：FSR3 路线是新的、也更贵。同一分辨率下实测 Resolve 约 57 ms/帧，FSR3 路线 84–205 ms/帧，所以默认仍是 Resolve。想试 FSR3 就把选项拨到 1，并确认上面那两个 `amd_fidelityfx_*.dll` 在位。

### 3. XeSS 帧生成：倍率扩到 2×–6×，非 Intel 显卡的多帧自动走兼容路径

- **帧倍率上限从 4× 提到 6×**（可选 2×–6×）。
- **2×**：走 SDK 原生路径，任何显卡都不需要额外处理。
- **3× 及以上**：在非 Intel 显卡上自动启用本项目核验过的兼容实现（内存内补丁 + 配套帧节奏修复），**不需要你手动开任何开关**。它要求两个文件都在位：`libxess_fg.dll` 必须是 **1.3.1.78** 那一个构建（SHA-256 以 `EC5E0C65E075570C…` 开头），以及它自己的依赖 **`libxell.dll`**——只放前者帧生成起不来。其他构建会被拒绝并说明原因。full 包这两个都带了。
- 3×/4× 现在也可以配 **NVIDIA 光流**（此前这个组合被直接拒绝）。

**要留意的**：这条多帧路径是本项目的实验性兼容实现，**不等于 Intel 官方对该组合的认证**。光流取自捕获画面、不是游戏引擎的原生运动矢量，所以平面深度、遮挡/UI/反射以及高倍率下的帧节奏等既有局限照旧；NVIDIA 高质量光流会更贵，**不保证提升最终显示帧率**。

### 4. 修复

- **抗闪烁的时域混合权重此前从未真正生效**——现在按预期混合了。
- 引导标志改在引擎构建 staging **之前**设置；Interop 不再被 pin 住。
- 进程定时器提到 1 ms：引擎内部的等待预算按 1 ms 计价，更粗的定时器会让它把正常等待读成超时。
- 重建路线的每个出口单独编号（原来都报同一条消息），每个会话打印落在哪个适配器上。

---

## 三、DLSSNR 面板怎么用

- **输入分辨率缩放**（25–100%）：网络按捕获尺寸的百分比运行，成本近似正比像素数，**这是唯一有效的提速手段**。1080p 级别的画面，缩放滑杆往下拉一档就能省下成比例的时间。
- **抗闪烁**：无 / 静态累积 / 光流累积 / 光流累积+ / 低频时域重建。画面稳定时用静态累积即可；有运动时用带光流的档位（用 Magpie 的 AMD 光流做重投影）。
- **NR Reconstruction**：见上，默认 Resolve。
- **NR Intensity**：编辑强度（0 = 不编辑，2 = 翻倍）。
- **NR Style**：三档预设（Default / Natural / Cinematic），对引擎的内容通道做整体缩放——Natural 更轻，适合二次元/赛璐璐；Cinematic 更强并打开色调通道。AMD 运行时没有自己的 style 字段，所以这三档缩放的是它实际有的通道。
- **tone / structure / skin / autoMask**，以及 Temporal History、Tone Channels、Use Depth Guide、Use Host Inputs：都直接映射到引擎自身设置。
- `uiCorrection` 无映射：本路径是抓成品画面，无法知道游戏 UI 在哪里。

高级设置（`dlssnr_on_amd.ini`）：

```ini
[DlssNrOnAmd]
SrgbInput=1      # 帧进引擎前 sRGB→线性、出来再编码回去。保持 1
Inline=1         # 同行握手：游戏在同一帧内等网络。本版实测就是这个模式
EditBound=500    # resolve 施加编辑量的千分比，高光溢出或闪烁时调低
Exposure=1.0     # 交给引擎的曝光（引擎自适应曝光不稳定，已停用）
```

---

## 四、实测与验证范围

- **DLSSNR 0.3.0 + Resolve 路线**，inline、输入输出零拷贝：**3600 帧长会话、等待超时 0 次**，网络约 **57 ms/帧**（1248×702 捕获、输入缩放 60% → 网络 749×421）；运动矢量真实（此前恒为 0.000）、depth on；引擎自检 `pre-block zero bytes 0.116% (healthy)`。
- **FSR3 重建路线**：可运行（日志 `route fsr`），但同分辨率下网络 84–205 ms，会话内出现 2 次 inline 等待超时并按设计降级为"显示上一帧残差"。因此默认仍是 Resolve。
- **XeSS 帧生成**：兼容路径的实测记录集中在 **3×/4×**；5×/6× 沿用同一条路径，但不在这次实测范围内。

---

## 五、已知限制

- inline 模式把帧率钉在网络的成本上（游戏等网络）；减小输入分辨率是唯一提速手段。
- FSR3 重建路线尚未做性能调优，明显慢于 Resolve。
- 引擎自身启动偶发崩溃（`bcrypt.dll+0x4442`，`0xc0000005`），不可稳定复现。
- GPL-3.0；本项目是独立的互操作工作，与 NVIDIA、AMD、Intel、DLSS-NR-on-AMD 均无隶属关系。

---

## English

Magpie (experimental 0.6.8) adapted to run DLSS Neural Rendering on AMD Radeon RDNA3 (tested on an RX 7900 XTX, gfx1100) with Cyberpunk 2077 and GTA V Enhanced. This is a patch on 0.6.8.

**Two packages:** `-full` is ready to use as it stands — it bundles the 0.3.0 runtime, the weights, the FSR3 upscaler DLLs, the XeSS frame-generation DLLs (`libxess_fg.dll` 1.3.1.78 plus its dependency `libxell.dll`), `libxess.dll` and the licences. Nothing has to be configured. The clean package contains only this port's own build products; it lists exactly which third-party files to supply and at which versions. The runtime and weights come from the third-party **OptiScaler AMD pre-SR** package, whose runtime is **DLSS-NR on AMD 0.3.0** (original author: danielblnc).

**What's new**

1. **DLSSNR now targets DLSS-NR-on-AMD 0.3.0.** The 2.21 and 0.3.0 layouts share no addresses, so the app verifies the runtime's SHA-256 (`8321cae7…`) and refuses a different build instead of following stale offsets. An older runtime is now rejected on purpose. 0.3.0 also needs 16 more bytes per frame (render extent and the colour's sub-pixel phase); without them the engine reads past the end and returns a black frame. **Update the runtime together with Magpie.**
2. **New effect parameter NR Reconstruction.** `0 Resolve (residual)` keeps the previous behaviour; `1 FSR3 upscale` hands the engine's edited frame to the FSR3 upscaler this build carries and reconstructs it to the display resolution. It is the newer and much more expensive route: measured ~57 ms/frame for Resolve versus 84–205 ms for FSR3 at the same extent, so Resolve stays the default.
3. **XeSS frame generation now goes up to 6×** (2×–6×, previously 2×–4×). At 2× every GPU uses the SDK's native path. At 3× and above, non-Intel GPUs automatically use this project's verified in-memory compatibility implementation — no toggle — which requires `libxess_fg.dll` build 1.3.1.78 (SHA-256 `EC5E0C65E075570C…`); other builds are rejected with a reason. 3×/4× can now also be combined with NVIDIA optical flow. This is an experimental compatibility path, not an Intel certification; optical flow comes from the captured image rather than the engine's own motion vectors, and the flat-depth, occlusion/UI/reflection and high-multiplier pacing limits still apply.
4. **Fixes:** the anti-flicker temporal blend weight was never actually applied; guide flags are now set before the engine builds its staging and Interop is no longer pinned; the process timer is 1 ms because that is the unit the engine's waits are priced in.

**Validated:** 0.3.0 with the Resolve route, inline, zero-copy — 3600 frames, zero wait timeouts, ~57 ms/frame at a 1248×702 capture scaled to 60%, real motion vectors, depth on. The FSR3 route runs but costs far more. The XeSS FG compatibility path's validation record covers 3×/4×; 5×/6× uses the same path but is outside this round's testing.

GPL-3.0. An independent interoperability effort; not affiliated with NVIDIA, AMD, Intel or the DLSS-NR-on-AMD project.
