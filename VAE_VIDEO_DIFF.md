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
