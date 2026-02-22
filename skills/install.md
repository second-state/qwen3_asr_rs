# Qwen3 ASR Skill — Installation Guide

Install the Qwen3 ASR skill for voice transcription using Qwen3-ASR-0.6B.

## Prerequisites

- `curl` (for downloading)
- `unzip` and `tar` (for extraction)
- `bash` (shell)
- `pip` with `huggingface_hub` and `transformers` (for model download and tokenizer generation)

## Quick Install (Recommended)

```bash
SKILL_DIR="${HOME}/.openclaw/skills/audio_asr"
mkdir -p "$SKILL_DIR"

# Clone the repo
git clone --depth 1 https://github.com/second-state/qwen3_asr_rs.git /tmp/qwen3-asr-repo
cp -r /tmp/qwen3-asr-repo/skills/* "$SKILL_DIR"
rm -rf /tmp/qwen3-asr-repo

# Download platform-specific binary, libtorch (Linux only), and model
"${SKILL_DIR}/bootstrap.sh"
```

After installation, verify it works:

```bash
# macOS (MLX backend — no extra env needed)
~/.openclaw/skills/audio_asr/scripts/asr \
  ~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B \
  /path/to/audio.wav

# Linux
LD_LIBRARY_PATH=~/.openclaw/skills/audio_asr/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  ~/.openclaw/skills/audio_asr/scripts/asr \
  ~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B \
  /path/to/audio.wav
```

## Manual Installation

If the automatic download fails, manually install the components:

1. Go to https://github.com/second-state/qwen3_asr_rs/releases/latest
2. Download the zip for your platform:
   - `asr-linux-x86_64.zip` (Linux x86_64)
   - `asr-linux-aarch64.zip` (Linux ARM64)
   - `asr-macos-aarch64.zip` (macOS Apple Silicon)
3. Extract the zip and copy the binary:
   ```bash
   mkdir -p ~/.openclaw/skills/audio_asr/scripts
   unzip asr-<platform>.zip
   cp asr-<platform>/asr ~/.openclaw/skills/audio_asr/scripts/asr
   chmod +x ~/.openclaw/skills/audio_asr/scripts/asr
   ```
4. **Linux only**: Download libtorch and extract to `~/.openclaw/skills/audio_asr/scripts/libtorch/`:
   - Linux x86_64: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
   - Linux aarch64: https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz

   macOS does not need libtorch — the binary uses the MLX backend natively.
5. Download model:
   ```bash
   huggingface-cli download Qwen/Qwen3-ASR-0.6B \
     --local-dir ~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B
   ```
6. Generate `tokenizer.json`:
   ```bash
   python3 -c "
   from transformers import AutoTokenizer
   import os
   path = os.path.expanduser('~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B')
   tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
   tok.backend_tokenizer.save(f'{path}/tokenizer.json')
   print(f'Saved {path}/tokenizer.json')
   "
   ```

## Troubleshooting

### Download Failed

Check network connectivity:

```bash
curl -I "https://github.com/second-state/qwen3_asr_rs/releases/latest"
```

### Unsupported Platform

Check your platform:

```bash
echo "OS: $(uname -s), Arch: $(uname -m)"
```

Supported: Linux (x86_64, aarch64) and macOS (Apple Silicon arm64).

### Missing libtorch (Linux only)

Ensure the library path includes the libtorch lib directory:

```bash
export LD_LIBRARY_PATH=~/.openclaw/skills/audio_asr/scripts/libtorch/lib:$LD_LIBRARY_PATH
```

The SKILL.md instructions set this automatically when running commands.
macOS does not need libtorch.
