# run_diff_video.py 结果说明

运行示例（输入视频为 `data/video/clip.mp4`）：

```bash
python run_diff_video.py --model_path /path/to/Wan2.2-TI2V-5B --video_tag clip
```

## 预处理视频

以下文件位于 `data/video/`，原视频保持不变：

| 文件 | 含义 |
| --- | --- |
| `clip_lowResolution.mp4` | 原视频缩放到高 384、宽 672，作为 A。 |
| `clip_first_frame.mp4` | 重复 A 的首帧，作为 B；帧数和帧率与 A 一致。 |

## 差值与生成结果

以下文件统一位于 `data/video/clip/`：

| 文件 | 含义 |
| --- | --- |
| `clip_lowResolution.pt` | A 的 VAE latent。 |
| `clip_first_frame.pt` | B 的 VAE latent。 |
| `A_reconstruction.mp4` | `decode(encode(A))`：判断正常动态视频的 VAE 重建是否闪烁。 |
| `B_reconstruction.mp4` | `decode(encode(B))`：判断静态首帧视频的 VAE 重建是否产生时间变化。 |
| `A-B.mp4` | `decode(encode(A) - encode(B))`，用于观察 latent 差值的解码效果。 |
| `diff_normalized.mp4` | 保留 A-B 视频中比中灰更暗的残差；灰色及亮侧变为白色，暗侧轮廓按整段视频的统一尺度归一化。 |
| `diff_normalized.pt` | 归一化残差图像的 VAE latent，从压缩前的图像直接编码。 |
| `first_frame_plus_diff_w0.mp4` | 首帧视频加噪后去噪，不注入残差。 |
| `first_frame_plus_diff_w0.1.mp4`、`first_frame_plus_diff_w0.3.mp4`、`first_frame_plus_diff_w1.mp4` | 首帧视频加噪后，分别加上 0.1、0.3、1 倍残差 latent，再去噪。 |
| `origin_minus_diff_w0.mp4` | 原视频加噪后去噪，不注入残差。 |
| `origin_minus_diff_w0.1.mp4`、`origin_minus_diff_w0.3.mp4`、`origin_minus_diff_w1.mp4` | 原视频加噪后，分别减去 0.1、0.3、1 倍残差 latent，再去噪。 |
| `metadata.json` | A/B 的路径、VAE 配置、尺寸、帧率和 latent 形状。 |
| `generation_metadata.json` | 生成参数、两路初始化公式、实际去噪步数和噪声强度。 |

两组对照共享同一份随机噪声、种子、提示词和加噪强度。初始化分别为：

- 首帧加残差：`(1-σ)*E(B) + σ*ε + w*E(R)`。
- 原视频减残差：`(1-σ)*E(A) + σ*ε - w*E(R)`。

R 仍为暗侧归一化后的残差视频。每组权重 w 为 0、0.1、0.3、1，在去噪前各注入一次。默认均从 50 步日程的中间开始，运行后 25 步；每次重新创建调度器，共生成 8 个对照视频。参数记录在 `generation_metadata.json`，其中 `residual_weight` 使用带正负号的实际系数。

旧的 `condition_noise_with_diff.mp4` 和随机噪声分支不再生成；已有旧文件不会自动删除，比较时请使用上述新文件名。

建议依次检查 A 重建、B 重建、权重 0 的基线，再对比 0.1/0.3/1。如果重建和基线稳定，而闪烁随权重增强，更支持残差注入导致或放大闪烁；如果正常重建已经闪烁，应先排查 VAE 阶段。不同内容的生成结果不能仅凭亮度变化就认定为异常，需结合运动观察。

输出视频采用 H.264/yuv420p，保持原帧数和帧率，尺寸为高 384、宽 672，不带音频。`.pt` 文件保存 CPU float32 Tensor，维度顺序为 `[C, T, H, W]`，可用 `torch.load(path, map_location="cpu", weights_only=True)` 读取。

## 如何理解画面

- `A-B.mp4` 可能出现灰色背景、固定首帧残影和移动轮廓；latent 差值解码不等于像素相减，也不是前景分割。
- `diff_normalized.mp4` 只按明暗保留单侧残差，没有透明通道；无法保证去掉同属暗侧的首帧残影。
- 两组最终视频用于比较首帧加残差与原视频减残差的生成效果。直接相加的残差 latent 不等于纯运动特征，不能保证准确复现原运动。

重新运行会覆盖对应结果文件。将示例中的 `clip` 替换为实际 `video_tag` 即可对应自己的输出。

注意：E(R) 是经过解码、单向归一化、再次编码后的表示，并不等于 E(A)-E(B)，所以首帧加残差不保证还原 A，原视频减残差也不保证还原 B。
