# Qwen3 ASR -- Rust CLI tools

Pure Rust implementation of [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) automatic speech recognition. The project builds a cross-platform CLI tool and API server suitable for agentic skills for AI agents and bots.

- **asr** generates text from an input audio file (supports most codex and file formats)
- **asr-server** runs an OpenAI-compatible HTTP API server for audio transcription

Supports two backends: **libtorch** (via the `tch` crate, cross-platform with optional CUDA) and **MLX** (Apple Silicon native via Metal GPU). Loads model weights directly from safetensors files and re-implements the complete neural network forward pass in Rust.

Learn more:
* [A Rust implementation / CLI](https://github.com/second-state/qwen3_tts_rs) for Qwen3's TTS (Text-to-Speech or speech synthesis) models
* An OpenAI compatible [API server for audio / speech](https://github.com/second-state/qwen3_audio_api/tree/main/rust)
* An OpenClaw SKILL for voice recognition. Copy and paste to your lobster to [install it](https://raw.githubusercontent.com/second-state/qwen3_asr_rs/refs/heads/main/skills/install.md)

## Quick Start

The install script automatically detects your platform (macOS/Linux/Windows, CPU/CUDA GPU), downloads the correct release binary, model weights, and a sample audio file:

### macOS / Linux

```bash
curl -sSf https://raw.githubusercontent.com/second-state/qwen3_asr_rs/main/install.sh | bash
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/second-state/qwen3_asr_rs/main/install.ps1 | iex
```

The installer will prompt you to choose a model size (0.6B recommended) and, on Linux with an NVIDIA GPU, whether to use CUDA or CPU.

Once complete, run your first transcription:

**macOS / Linux:**

```bash
cd qwen3_asr_rs
./asr ./Qwen3-ASR-0.6B sample.wav
```

**Windows:**

```powershell
cd qwen3_asr_rs
.\asr .\Qwen3-ASR-0.6B sample.wav
```

Output:

```
Language: English
Text: Thank you for your contribution to the most recent issue of Computer.
```

## Architecture

The implementation ports the Qwen3-ASR encoder-decoder architecture from PyTorch/Transformers to Rust with libtorch (via the `tch` crate):

- **Audio Encoder** (Whisper-style): 3x Conv2d downsampling → sinusoidal positional embeddings → 18 transformer encoder layers → output projection (896 → 1024)
- **Text Decoder** (Qwen3): 28 transformer decoder layers with Grouped Query Attention (16 Q heads / 8 KV heads), QK-normalization, MRoPE (Multimodal Rotary Position Embeddings), and SwiGLU MLP
- **Audio preprocessing**: FFmpeg decodes any audio format → resampled to mono 16kHz f32 → 128-bin log-mel spectrogram (Whisper-style)

## Supported Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| Qwen3-ASR-0.6B | 0.6B | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| Qwen3-ASR-1.7B | 1.7B | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |

## Usage

```bash
# Basic transcription (auto-detect language)
asr ./Qwen3-ASR-0.6B input.wav

# Force language
asr ./Qwen3-ASR-0.6B input.wav chinese
asr ./Qwen3-ASR-0.6B input.wav english

# Enable debug logging
RUST_LOG=debug asr ./Qwen3-ASR-0.6B input.wav
```

### Output Format

```
Language: Chinese
Text: 你好世界
```

## API Server

The `asr-server` binary provides an OpenAI-compatible HTTP API for audio transcription.

### Start the Server

```bash
asr-server --model-dir ./Qwen3-ASR-0.6B
```

Options:

```
--model-dir <PATH>      Path to the Qwen3-ASR model directory (required)
--host <ADDR>           Host address to bind to (default: 0.0.0.0)
--port <PORT>           Port to listen on (default: 8080)
--language <LANG>       Default language for transcription (e.g., chinese, english)
-v, -vv                 Verbose output (debug, trace)
```

### Endpoints

#### POST /v1/audio/transcriptions

OpenAI-compatible transcription endpoint. Accepts multipart form data.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | binary | Yes | Audio file (any format supported by FFmpeg) |
| `language` | string | No | Language hint (e.g., `english`, `chinese`) |
| `response_format` | string | No | `json` (default), `text`, or `verbose_json` |
| `model` | string | No | Accepted for compatibility, ignored |
| `temperature` | float | No | Accepted for compatibility, ignored |
| `prompt` | string | No | Accepted for compatibility, ignored |

Examples:

```bash
# JSON response (default)
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav

# {"text":"Thank you for your contribution..."}

# Plain text response
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F response_format=text

# Thank you for your contribution...

# Verbose JSON with language and duration
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F response_format=verbose_json

# {"task":"transcribe","language":"English","duration":7.999,"text":"Thank you..."}

# Force language
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F language=chinese
```

#### GET /v1/models

Lists available models.

```bash
curl http://localhost:8080/v1/models
# {"object":"list","data":[{"id":"qwen3-asr","object":"model","owned_by":"qwen"}]}
```

#### GET /health

Health check endpoint.

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

## Supported Languages

Qwen3-ASR supports 30 languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

## Build from Source

### Prerequisites

Download model weights and generate the tokenizer:

```bash
pip install huggingface_hub transformers

huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir Qwen3-ASR-0.6B

python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen3-ASR-0.6B', trust_remote_code=True)
tok.backend_tokenizer.save('Qwen3-ASR-0.6B/tokenizer.json')
"
```

### Build for macOS (MLX)

```bash
git submodule update --init --recursive
cargo build --release --no-default-features --features mlx
```

### Build for Linux (libtorch)

Download and extract libtorch for your platform from [libtorch-releases](https://github.com/second-state/libtorch-releases/releases/tag/v2.7.1):

```bash
# Linux x86_64 (CPU)
curl -LO https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-x86_64-2.7.1.tar.gz
tar xzf libtorch-cxx11-abi-x86_64-2.7.1.tar.gz

# Linux x86_64 (CUDA 12.6)
curl -LO https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-x86_64-cuda12.6-2.7.1.tar.gz
tar xzf libtorch-cxx11-abi-x86_64-cuda12.6-2.7.1.tar.gz

# Linux ARM64 (CPU)
curl -LO https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz
tar xzf libtorch-cxx11-abi-aarch64-2.7.1.tar.gz

# Linux ARM64 (CUDA 12.6 / Jetson)
curl -LO https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-cuda12.6-2.7.1.tar.gz
tar xzf libtorch-cxx11-abi-aarch64-cuda12.6-2.7.1.tar.gz
```

Set environment variables:

```bash
export LIBTORCH=$(pwd)/libtorch
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

Install dependencies and build:

```bash
cargo build --release
```

## Project Structure

```
src/
├── main.rs            # CLI binary entry point
├── bin/
│   └── server.rs      # API server binary entry point
├── lib.rs             # Library module declarations
├── tensor.rs          # Unified Tensor abstraction (tch/MLX backend)
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
├── inference.rs       # End-to-end ASR inference pipeline
└── backend/
    └── mlx/           # Apple MLX backend (Metal GPU)
        ├── ffi.rs     # Raw C FFI bindings to mlx-c
        ├── array.rs   # Safe RAII MlxArray wrapper
        ├── ops.rs     # Safe operation wrappers
        ├── io.rs      # Safetensors loading via mlx-c
        ├── signal.rs  # STFT, mel spectrogram signal processing
        └── stream.rs  # Device/stream management
```

## Performance

Benchmarked on Apple M4 Mac Mini (16GB RAM), MLX Metal GPU backend. All times are warm runs (post-shader compilation), best-of-3.

### Qwen3-ASR-0.6B

| Test File | Audio Duration | Tokens | CLI | API Server |
|-----------|---------------|--------|-----|------------|
| sample1.wav (English) | 8.0s | 31 | 2.35s | 2.10s |
| speech_en.wav (English) | 3.5s | 15 | 1.30s | 1.05s |
| sample2.wav (English) | 2.8s | 13 | 1.17s | 0.95s |
| sample3.wav (Chinese) | 5.6s | 15 | 1.31s | 1.07s |

### Qwen3-ASR-1.7B

| Test File | Audio Duration | Tokens | CLI | API Server |
|-----------|---------------|--------|-----|------------|
| sample1.wav (English) | 8.0s | 31 | 6.26s | 5.80s |
| speech_en.wav (English) | 3.5s | 15 | 3.40s | 3.06s |
| sample2.wav (English) | 2.8s | 13 | 2.82s | 2.59s |
| sample3.wav (Chinese) | 5.6s | 15 | 3.31s | 2.94s |

The API server is faster per request because the model stays loaded in memory, avoiding the process startup and model loading overhead of the CLI.

## License

Apache-2.0
