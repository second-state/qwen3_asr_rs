# Qwen3 ASR Skill — Installation Guide

Install the Qwen3 ASR skill for voice transcription using Qwen3-ASR-0.6B and Qwen3-ASR-1.7B.

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

# Download platform-specific binary, libtorch, and models
"${SKILL_DIR}/bootstrap.sh"
```

After installation, verify it works:

```bash
# Linux
LD_LIBRARY_PATH=~/.openclaw/skills/audio_asr/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  ~/.openclaw/skills/audio_asr/scripts/asr \
  ~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B \
  /path/to/audio.wav

# macOS
DYLD_LIBRARY_PATH=~/.openclaw/skills/audio_asr/scripts/libtorch/lib:$DYLD_LIBRARY_PATH \
  ~/.openclaw/skills/audio_asr/scripts/asr \
  ~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B \
  /path/to/audio.wav
```

## Manual Installation

If the automatic download fails, manually install the components:

1. Go to https://github.com/second-state/qwen3_asr_rs/releases/latest
2. Download the binary for your platform:
   - `asr-linux-x86_64` (Linux x86_64)
   - `asr-linux-aarch64` (Linux ARM64)
   - `asr-macos-aarch64` (macOS Apple Silicon)
3. Copy to `~/.openclaw/skills/audio_asr/scripts/asr` and make it executable:
   ```bash
   mkdir -p ~/.openclaw/skills/audio_asr/scripts
   cp asr-<platform> ~/.openclaw/skills/audio_asr/scripts/asr
   chmod +x ~/.openclaw/skills/audio_asr/scripts/asr
   ```
4. Download libtorch and extract to `~/.openclaw/skills/audio_asr/scripts/libtorch/`:
   - Linux x86_64: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
   - Linux aarch64: https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz
   - macOS arm64: https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip
5. Download models:
   ```bash
   huggingface-cli download Qwen/Qwen3-ASR-0.6B \
     --local-dir ~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B
   huggingface-cli download Qwen/Qwen3-ASR-1.7B \
     --local-dir ~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-1.7B
   ```
6. Generate `tokenizer.json` for each model:
   ```bash
   python3 -c "
   from transformers import AutoTokenizer
   import os
   base = os.path.expanduser('~/.openclaw/skills/audio_asr/scripts/models')
   for model in ['Qwen3-ASR-0.6B', 'Qwen3-ASR-1.7B']:
       path = f'{base}/{model}'
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

### Missing libtorch

Ensure the library path includes the libtorch lib directory:

- Linux: `export LD_LIBRARY_PATH=~/.openclaw/skills/audio_asr/scripts/libtorch/lib:$LD_LIBRARY_PATH`
- macOS: `export DYLD_LIBRARY_PATH=~/.openclaw/skills/audio_asr/scripts/libtorch/lib:$DYLD_LIBRARY_PATH`

The SKILL.md instructions set this automatically when running commands.
