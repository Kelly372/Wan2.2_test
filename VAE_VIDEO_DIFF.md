# 视频 latent 差值实验

将原视频放在 `data/video/clip.mp4`，先生成首帧视频 B，再编码 A/B 并解码差值：

```bash
python prepare_videos.py --video_tag clip
python vae_video_diff.py --video_tag clip --vae_checkpoint /path/to/Wan2.2_VAE.pth
```

默认使用 Wan2.2 VAE（TI2V-5B 对应的 VAE）和 CUDA。若使用 A14B 模型对应的 Wan2.1 VAE：

```bash
python vae_video_diff.py --video_tag clip --vae_type 2.1 --vae_checkpoint /path/to/Wan2.1_VAE.pth
```

`--vae_checkpoint` 指向 VAE 权重文件，而非模型目录。`--device cpu` 可使用 CPU；默认以 float32 计算。脚本仅加载 VAE，需要 `torch`、`einops`、`opencv-python`、`numpy` 和 `imageio-ffmpeg`，不需要加载扩散模型。

首帧视频和差值视频统一保存为 MP4 / H.264（`libx264`），像素格式为 `yuv420p`，使用 CRF 18 和 `faststart`。编码直接调用 FFmpeg，不依赖 OpenCV 是否编译了 H.264 编码器。按帧写入，保持输入尺寸和帧率，不自动缩放到 16 的倍数。

Linux 环境安装仓库依赖后即可运行。若已有环境缺少编码依赖，执行 `python -m pip install imageio-ffmpeg`。该包通常附带 FFmpeg；也可以通过 `export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg` 指定具有 `libx264` 编码器的系统 FFmpeg。

修改代码不会转换旧视频；重新运行上述两个命令可覆盖生成兼容格式的视频及相应 latent 文件。

所有数据路径相对于脚本所在的仓库根目录。只需要原视频，首帧视频保持原视频的帧数、分辨率和帧率。

| 输入 | 保存的 latent |
| --- | --- |
| `data/video/clip.mp4`（A） | `data/video/clip/clip.pt` |
| `data/video/clip_first_frame.mp4`（B） | `data/video/clip/clip_first_frame.pt` |

每个 `.pt` 直接保存一个 CPU float32 Tensor，形状为 `[C, T, H, W]`，使用 `torch.load(path, map_location="cpu", weights_only=True)` 读取。它是仓库 VAE `encode()` 返回的归一化 latent。每次运行重新编码并覆盖这两个文件。

latent 和差值实验结果统一保存在 `data/video/clip/`：

- `A-B.mp4`：`decode(encode(A) - encode(B))`
- `clip.pt`、`clip_first_frame.pt`：A/B 的 latent。
- `metadata.json`：输入路径、权重路径、帧率、原始尺寸和 latent 形状等实验信息。

A/B 必须具有相同的帧数、分辨率和帧率，否则脚本报错。重新运行第一步可生成与当前原视频匹配的 B。

编码前将最后一帧重复补齐到 `4n+1` 帧，并用边缘像素向右、向下补齐到 VAE 的空间步长（2.1 为 8，2.2 为 16）。解码后裁回原视频帧数和尺寸，沿用原帧率，不带音频。MP4 输出要求原宽高为偶数。

这里直接对归一化 latent 相减，再调用仓库的 `decode()`，不额外调整均值或缩放差值。VAE 是非线性的，因此输出是 latent 差值的解码实验，并不等同于像素相减或前景分割。解码器的 `[-1, 1]` 输出线性映射到 `[0, 255]` 后保存。

## 残差注入去噪生成

完成上述实验、得到 `data/video/clip/A-B.mp4` 后，运行：

```bash
export WAN_CKPT_DIR=/path/to/Wan2.2-TI2V-5B
python generate_with_diff.py --video_tag clip
```

也可把完整模型放在仓库根目录 `Wan2.2-TI2V-5B/`，或通过 `--ckpt_dir` 指定目录。此步骤需要完整 TI2V-5B 权重（DiT、T5、tokenizer、Wan2.2 VAE）和 CUDA GPU；仅有 VAE 权重不能去噪生成。目前接入的是 TI2V-5B 的无首帧约束生成路径，不支持 A14B 双专家模型。使用仓库完整推理依赖。

对读入的残差视频 RGB 值 `I`，默认计算 `u=max(127.5-I, 0)`，然后以整段视频共享的最大值归一化为 `R=255*(1-u/max(u))`。灰色和亮侧映射为白色，暗侧保留为深色轮廓；无暗侧残差时明确报错。此操作没有透明通道，也不使用空间分割，不能保证去除与移动轮廓同属暗侧的首帧残影。`--baseline` 可调整灰色基准；全局归一化保留帧间相对强弱，避免逐帧自动拉伸。

对归一化图像 R 和原视频重新进行 VAE 编码，统一补齐到 `4n+1` 帧和 32 的空间倍数（满足 VAE 及 DiT patch 要求）。不会复用之前可能采用不同补齐尺寸或 VAE 版本的 `.pt`。R 直接从内存编码，保存的 MP4 仅用于查看，避免额外有损压缩后再编码。

令 `z_R=E(R)`、`z_A=E(A)`，两路共享同一份标准高斯噪声 `ε`，在去噪开始前各执行一次不带额外权重的逐元素加法：

- 随机分支：`x_start=ε+z_R`，从完整噪声日程开始去噪。
- 原视频分支：`x_start=(1-σ)*z_A+σ*ε+z_R`，从对应 σ 的日程位置开始去噪。使用 Wan flow-matching 的加噪公式。

默认 `--inference_step 50`，使用官方 TI2V-5B 的 UniPC、shift=5、CFG=5。新增实验参数 `--strength 0.5` 表示原视频分支使用后半段日程，因此默认随机分支运行 50 步、原视频分支运行 25 步；strength 并非实际 σ，实际 σ 会记录到结果。strength=1 时原视频信息基本消失，两路初始化趋于一致。默认 `--seed 42`、`--prompt ""`，两路使用同一文本条件及官方负面提示词，可指定共同的内容描述。整个去噪循环不再重复注入残差。

以下结果保存到 `data/video/clip/`，生成视频裁回原尺寸和帧数，保持原帧率，采用 H.264/yuv420p：

- `diff_normalized.mp4`：单向归一化残差预览。
- `diff_normalized.pt`：归一化残差的 VAE latent。
- `random_noise_with_diff.mp4`：随机噪声加残差 latent 后去噪生成。
- `condition_noise_with_diff.mp4`：原视频加噪后加残差 latent，再去噪生成。
- `generation_metadata.json`：完整参数、初始化公式、各路实际步数和噪声强度。

这是直接 latent 相加实验；归一化残差的编码包含白色背景的表示，也不等价于纯运动向量。它会改变标准噪声分布，模型没有专门针对这种注入进行训练，生成质量和首帧残影去除程度需要实测判断。
