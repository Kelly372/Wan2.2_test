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
| `A-B.mp4` | `decode(encode(A) - encode(B))`，用于观察 latent 差值的解码效果。 |
| `diff_normalized.mp4` | 保留 A-B 视频中比中灰更暗的残差；灰色及亮侧变为白色，暗侧轮廓按整段视频的统一尺度归一化。 |
| `diff_normalized.pt` | 归一化残差图像的 VAE latent，从压缩前的图像直接编码。 |
| `random_noise_with_diff.mp4` | 随机噪声加上归一化残差 latent，再去噪生成。 |
| `condition_noise_with_diff.mp4` | A 的 latent 加噪后，再加上归一化残差 latent，随后去噪生成。 |
| `metadata.json` | A/B 的路径、VAE 配置、尺寸、帧率和 latent 形状。 |
| `generation_metadata.json` | 生成参数、两路初始化公式、实际去噪步数和噪声强度。 |

两路使用同一份随机噪声，并在去噪前各执行一次逐元素加法。默认随机分支运行 50 步；原视频分支从日程中间开始，运行后 25 步，以保留部分原视频信息。

输出视频采用 H.264/yuv420p，保持原帧数和帧率，尺寸为高 384、宽 672，不带音频。`.pt` 文件保存 CPU float32 Tensor，维度顺序为 `[C, T, H, W]`，可用 `torch.load(path, map_location="cpu", weights_only=True)` 读取。

## 如何理解画面

- `A-B.mp4` 可能出现灰色背景、固定首帧残影和移动轮廓；latent 差值解码不等于像素相减，也不是前景分割。
- `diff_normalized.mp4` 只按明暗保留单侧残差，没有透明通道；无法保证去掉同属暗侧的首帧残影。
- 两路最终视频用于比较原视频初始化和随机初始化下的生成效果。直接相加的残差 latent 不等于纯运动特征，不能保证准确复现原运动。

重新运行会覆盖对应结果文件。将示例中的 `clip` 替换为实际 `video_tag` 即可对应自己的输出。
