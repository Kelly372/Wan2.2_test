# run_layer_video.py：固定后五步的层分组实验

```bash
python run_layer_video.py --model_path /path/to/Wan2.2-TI2V-5B --video_tag clip
```

输入为 `data/video/clip.mp4`。脚本复用预处理，将 A 缩放为 H=384、W=672，生成首帧重复视频 B，保留输入帧数和 FPS。完整模型目录包含 `Wan2.2_VAE.pth`，需要仓库推理依赖和 CUDA。

默认运行前、中、后三组，每组分别执行 A 接收 B 和 B 接收 A，共 6 个实验视频，加 2 个未交换基线。可加 `--layer_mode front`、`--layer_mode middle` 或 `--layer_mode back` 单独运行一组；默认 `all`。

## 固定条件

- **层编号从 0 开始**：front=0–9、middle=10–19、back=20–29，均包含端点。
- **更新编号从 1 开始**：仅在实际第 21、22、23、24、25 次去噪更新中持续替换；前 20 次正常运行。不是原始 timestep 数值 21–25，也不是旧文件名的剩余步数。
- 沿用 `run_replace_video.py` 的 50 步 UniPC 调度后 25 步，替换对应原调度索引 45–49。初始 σ 约 0.833055，保持 seed=42、空正向提示词、官方负面提示词、shift=5、CFG=5。
- 初始 latent 为 `(1-σ)*E(A/B)+σ*同一份噪声`，无原始残差注入；不切换到 σ=0.2 或原生 I2V，以便只比较层范围。
- 供体每一步都来自独立、未受交换影响的同步基线。条件与无条件 CFG 分支分别替换对应特征。
- 替换位置是所选层 `block.self_attn` 的投影后输出。组外层正常执行，仍可能受到前面层改变的隐藏状态影响；接收方的残差连接、cross-attention、FFN、latent 和求解器历史保留。

## 输出

新结果保存到 `data/video/clip/attention_layers/`，与旧单步和连续窗口实验区分：

| 文件 | 含义 |
| --- | --- |
| `layer_process_A_baseline.mp4`、`layer_process_B_baseline.mp4` | 未交换基线 |
| `layer_front_00-09_swap_A_from_B_updates21-25.mp4` | A 接收 B 的前 10 层输出 |
| `layer_middle_10-19_swap_A_from_B_updates21-25.mp4` | A 接收 B 的中间 10 层输出 |
| `layer_back_20-29_swap_A_from_B_updates21-25.mp4` | A 接收 B 的后 10 层输出 |
| 上述文件名中的 `A_from_B` 改为 `B_from_A` | 反向交换 |

每个实验附同名前缀的 `_diagnostics.json`，定义沿用 `run_replace_video.py`：干预步记录条件/无条件/CFG 预测差异、复制求解器历史得到的局部更新差异；全部 25 步记录相对原接收方基线的累计偏离。数值包括 RMSE、最大绝对差、参考 RMS、相对 L2。

`layer_metadata.json` 记录本次实验模式、层编号、更新编号、实际 timestep/σ、采样设置和基线自身替换检查。重跑同一模式会覆盖本目录内对应结果，旧 `attention_extended/` 不受影响。诊断在每条轨迹完成时写入；视频在全部选中轨迹完成后统一解码。

## 如何比较

先与本次 A/B 基线比较，再与此前 `window_swap_*_updates21-25.mp4` 的全 30 层结果比较。如果某组控制明显，说明这组层在该窗口对当前干预更敏感，不能直接认定其独立负责相机控制。若三组都弱于全层交换，也可能存在跨层协同。

同时看运动方向和幅度、边缘清晰度、闪烁与形变。仅有大 latent 差异不等于有效视角控制。

共享工具只增加了可选层范围参数；`run_replace_video.py` 默认仍替换全部层。CPU 测试检查组外层正常计算、三组自身替换、同步供体、CFG 双分支、后五步替换和旧流程回归。GPU 实际生成与画质仍需在有模型的环境验证。
