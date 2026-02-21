# Qwen3 ASR Rust

Pure Rust implementation of [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) automatic speech recognition using libtorch. Loads model weights directly from safetensors files and re-implements the complete neural network forward pass in Rust.

## Architecture

The implementation ports the Qwen3-ASR encoder-decoder architecture from PyTorch/Transformers to Rust with libtorch (via the `tch` crate):

- **Audio Encoder** (Whisper-style): 3x Conv2d downsampling → sinusoidal positional embeddings → 18 transformer encoder layers → output projection (896 → 1024)
- **Text Decoder** (Qwen3): 28 transformer decoder layers with Grouped Query Attention (16 Q heads / 8 KV heads), QK-normalization, MRoPE (Multimodal Rotary Position Embeddings), and SwiGLU MLP
- **Audio preprocessing**: FFmpeg (statically linked) decodes any audio format → resampled to mono 24kHz 16-bit PCM → internally resampled to 16kHz → 128-bin log-mel spectrogram (Whisper-style)

## Supported Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| Qwen3-ASR-0.6B | 0.6B | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |

## Prerequisites

### libtorch

Download and extract libtorch for your platform:

```bash
# macOS (Apple Silicon)
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.10.0.zip
unzip libtorch-macos-arm64-2.10.0.zip

# Linux (CPU)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.10.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.10.0+cpu.zip

# Linux (CUDA 12.6)
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.10.0%2Bcu126.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.10.0+cu126.zip
```

Or symlink to an existing libtorch installation:

```bash
ln -s /path/to/libtorch ./libtorch
```

### FFmpeg

Install FFmpeg development libraries:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libswresample-dev pkg-config
```

### Model Weights

Download the model from HuggingFace:

```bash
# Using git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-ASR-0.6B

# Or using huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir Qwen3-ASR-0.6B
```

Generate `tokenizer.json` (required):

```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen3-ASR-0.6B', trust_remote_code=True)
tok.backend_tokenizer.save('Qwen3-ASR-0.6B/tokenizer.json')
"
```

## Build

```bash
# Set environment
export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH    # Linux
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH  # macOS

# Build (dynamically links FFmpeg)
cargo build --release

# Build with statically linked FFmpeg
cargo build --release --features static-ffmpeg

# Build FFmpeg from source and link statically (most self-contained)
cargo build --release --features build-ffmpeg
```

## Usage

```bash
# Basic transcription (auto-detect language)
asr ./Qwen3-ASR-0.6B input.wav

# Force language
asr ./Qwen3-ASR-0.6B input.wav chinese
asr ./Qwen3-ASR-0.6B input.wav english

# Any audio format (FFmpeg handles conversion)
asr ./Qwen3-ASR-0.6B input.mp3
asr ./Qwen3-ASR-0.6B input.flac
asr ./Qwen3-ASR-0.6B input.m4a

# Enable debug logging
RUST_LOG=debug asr ./Qwen3-ASR-0.6B input.wav
```

### Input Audio Requirements

The `asr` binary accepts **any audio format** supported by FFmpeg. The audio is automatically preprocessed to:

- Mono channel
- 24 kHz sample rate
- 16-bit signed PCM

Then internally resampled to 16 kHz for the model's mel spectrogram computation.

### Output Format

```
Language: Chinese
Text: 你好世界
```

## Supported Languages

Qwen3-ASR supports 30 languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

## Project Structure

```
src/
├── main.rs            # CLI binary entry point
├── lib.rs             # Library module declarations
├── config.rs          # Model configuration (from config.json)
├── error.rs           # Error types
├── audio.rs           # FFmpeg-based audio loading and format conversion
├── mel.rs             # Whisper-style mel spectrogram feature extraction
├── weights.rs         # Safetensors weight loading (bf16 → f32 conversion)
├── layers.rs          # Neural network building blocks (LayerNorm, RMSNorm,
│                      #   attention, MLP, MRoPE, etc.)
├── audio_encoder.rs   # Whisper-style audio encoder (Conv2d + Transformer)
├── text_decoder.rs    # Qwen3 text decoder with KV cache
├── tokenizer.rs       # HuggingFace tokenizer wrapper
└── inference.rs       # End-to-end ASR inference pipeline
```

## License

Apache-2.0
