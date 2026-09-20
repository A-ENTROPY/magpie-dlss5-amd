# 大力喜鹊 AMD RDNA3 v0.6.9 — DLSSNR 0.3.0 契约迁移与链内重建

大力喜鹊（Magpie experimental）的 **AMD Radeon RDNA3 适配版**，在 RX 7900 XTX（gfx1100）上运行 DLSS Neural Rendering。本版把 AMD 后端的运行时契约从 2.21 迁到 **DLSS-NR-on-AMD 0.3.0**，并新增链内 FSR3 重建路线。

## 两个包，按需取用

- **`magpie-dlss5-amd-rdna3-full.zip`（开箱即用）**：完整应用 + 0.3.0 运行时 + 网络权重 + FSR3 上采样 DLL，解压即用。注意：运行时与权重来自第三方项目与 NVIDIA 派生材料，仅随本包提供、不在此授权进一步转发。
- **`magpie-dlss5-amd-rdna3.zip`（合规干净包）**：仅本适配的构建产物。运行时与权重需自行从 DLSS-NR-on-AMD 官方安装器取得；想用 FSR3 重建路线还需自备 `amd_fidelityfx_upscaler_dx12.dll` 与 `amd_fidelityfx_loader_dx12.dll`。

## 本版更新

**运行时契约迁移到 0.3.0。** AMD runtime 换了布局，0.2.14（`3c9ca13f…`）与 0.3.0（`8321cae7…`）**没有任何地址相同**，因此后端现在按 SHA-256 校验运行时：不匹配就拒绝加载，而不是按旧偏移跳进未知布局。0.3.0 的 packet 比旧版多 16 字节（`nativePre` 与 render extent / 子像素相位），少发这 16 字节不是"差一点"——引擎会读越界并交回黑帧。

**新增 NR Reconstruction（效果参数 `amdReconstruct`）：**

- `0 Resolve (residual)`（默认）：引擎编辑后的帧按残差就地 resolve。
- `1 FSR3 upscale`：把引擎的输出交给链内的 FSR3 上采样器重建到显示分辨率，并接上引擎的 HDR 协议与运动引导。

**修复与调优：**

- 时域混合权重此前从未真正施加（`decf4a94`）——抗闪烁路线现在按预期混合。
- guide flags 改为在引擎构建 staging **之前**设置，Interop 不再被 pin。
- 进程定时器提到 1 ms：引擎内部的等待预算按 1 ms 计价，2 ms 及以上会让它把等待读成超时。
- 诊断：每个会话打印落在哪个适配器上；重建路线的每个出口单独编号，失败不再都报同一条消息。

## 验证记录

- **0.3.0 + Resolve 路线**（inline、输入输出均零拷贝）：3600 帧长会话，`timeouts 0`，网络 ~57 ms/帧（1248×702 捕获、输入缩放 0.6 → 网络 749×421）；运动矢量真实（`mean |mv|` 非零，此前恒为 0.000），depth on；引擎自检 `pre-block zero bytes 0.116% (healthy)`。
- **FSR3 路线**实测可运行（日志 `route fsr`），但同分辨率下网络耗时显著更高（84–205 ms，会话内有 2 次 inline 等待超时并按设计降级为"显示上一帧残差"）。因此默认仍是 Resolve，FSR3 路线供愿意换成本的用户试用。
- 两轮会话都跑在 1248×702 捕获、输入分辨率 60% 上；成本近似正比于像素数，减小输入分辨率仍是唯一提速手段。

## 已知限制

- inline 模式把帧率钉在网络的成本上（游戏等网络）；减小输入分辨率是唯一提速手段。
- FSR3 重建路线尚未做性能调优，明显慢于 Resolve。
- 引擎自身启动偶发崩溃（`bcrypt.dll+0x4442`，`0xc0000005`），不可稳定复现。
- `uiCorrection` 无映射（本路径无法知道游戏 UI 在哪）；AMD runtime 没有 style 字段，NR Style 三档缩放的是它实际有的内容通道。
- GPL-3.0；与 NVIDIA、AMD、DLSS-NR-on-AMD 均无隶属关系。

---

## English

Magpie (experimental) AMD RDNA3 adaptation for DLSS Neural Rendering on a Radeon RX 7900 XTX (gfx1100). This release moves the AMD backend's runtime contract from 2.21 to **DLSS-NR-on-AMD 0.3.0** and adds an in-chain FSR3 reconstruction route.

- The two runtime layouts share no addresses, so the backend now verifies the runtime's SHA-256 and refuses a mismatch instead of chasing stale offsets. The 0.3.0 packet is 16 bytes longer (render extent and sub-pixel phase); sending the shorter one makes the engine read past the end and return a black frame.
- New effect parameter **NR Reconstruction** (`amdReconstruct`): `0 Resolve (residual)` (default) or `1 FSR3 upscale`, which hands the engine's edited frame to the in-chain FSR3 upscaler, with the engine's HDR protocol and a motion guide.
- Fixes: the temporal blend weight was never applied; guide flags are now set before the engine builds its staging and Interop is no longer pinned; the process timer is 1 ms because that is what the engine's waits are priced in.
- Validated: 0.3.0 with the Resolve route, inline, zero-copy, 3600 frames, zero wait timeouts, ~57 ms/frame at a 1248×702 capture scaled to 0.6, real motion vectors, depth on. The FSR3 route runs but costs far more at the same extent (84–205 ms, two wait timeouts), so Resolve stays the default.
- GPL-3.0. Independent interoperability effort; not affiliated with NVIDIA, AMD or the DLSS-NR-on-AMD project.
