#!/bin/bash
# install.sh — One-step installer for Qwen3-ASR Rust CLI
# Downloads the release binary (with bundled libtorch), model weights, and a sample audio file.

set -e

REPO="second-state/qwen3_asr_rs"
INSTALL_DIR="qwen3_asr_rs"
SAMPLE_WAV_URL="https://github.com/${REPO}/raw/main/test_audio/sample1.wav"

# ── colours / helpers ────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[error]${NC} $*" >&2; }

# ── 1. Detect platform ──────────────────────────────────────────────
detect_platform() {
    local os arch cuda=""

    case "$(uname -s)" in
        Linux*)  os="linux"  ;;
        Darwin*) os="macos"  ;;
        *)
            err "Unsupported OS: $(uname -s)"
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)    arch="x86_64"  ;;
        aarch64|arm64)   arch="aarch64" ;;
        *)
            err "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac

    # CUDA detection (Linux only — macOS uses Metal via MLX)
    if [ "$os" = "linux" ]; then
        if command -v nvidia-smi &>/dev/null; then
            cuda=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
        fi
    fi

    OS="$os"
    ARCH="$arch"
    CUDA_DRIVER="$cuda"
}

print_platform() {
    echo ""
    info "System detection"
    echo "  OS:           ${OS}"
    echo "  CPU:          ${ARCH}"

    if [ "$OS" = "macos" ]; then
        echo "  GPU:          Apple Silicon (Metal via MLX)"
    elif [ -n "$CUDA_DRIVER" ]; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
        echo "  GPU:          ${gpu_name} (CUDA driver ${CUDA_DRIVER})"
    else
        echo "  GPU:          None detected"
    fi
    echo ""
}

# ── 2. Map platform → release asset ─────────────────────────────────
resolve_asset() {
    case "${OS}-${ARCH}" in
        macos-aarch64)  ASSET_NAME="asr-macos-aarch64"  ;;
        linux-x86_64)
            if [ -n "$CUDA_DRIVER" ]; then
                echo ""
                info "NVIDIA GPU detected. Choose build variant:"
                echo "  1) CUDA  (recommended for GPU)"
                echo "  2) CPU only"
                echo ""

                local choice
                read -r -p "Select variant [1]: " choice </dev/tty
                choice="${choice:-1}"

                case "$choice" in
                    1) ASSET_NAME="asr-linux-x86_64-cuda" ;;
                    2) ASSET_NAME="asr-linux-x86_64" ;;
                    *)
                        warn "Invalid choice '${choice}', defaulting to CUDA."
                        ASSET_NAME="asr-linux-x86_64-cuda"
                        ;;
                esac
            else
                ASSET_NAME="asr-linux-x86_64"
            fi
            ;;
        linux-aarch64)
            if [ -n "$CUDA_DRIVER" ]; then
                echo ""
                info "NVIDIA GPU detected on ARM64 (Jetson). Choose build variant:"
                echo "  1) CUDA  (recommended for Jetson)"
                echo "  2) CPU only"
                echo ""

                local choice
                read -r -p "Select variant [1]: " choice </dev/tty
                choice="${choice:-1}"

                case "$choice" in
                    1) ASSET_NAME="asr-linux-aarch64-cuda" ;;
                    2) ASSET_NAME="asr-linux-aarch64" ;;
                    *)
                        warn "Invalid choice '${choice}', defaulting to CUDA."
                        ASSET_NAME="asr-linux-aarch64-cuda"
                        ;;
                esac
            else
                ASSET_NAME="asr-linux-aarch64"
            fi
            ;;
        macos-x86_64)
            err "macOS x86_64 (Intel) is not supported. Apple Silicon required."
            exit 1
            ;;
        *)
            err "No pre-built release for ${OS}-${ARCH}."
            exit 1
            ;;
    esac
}

# ── 3. Download & extract release ────────────────────────────────────
# All Linux release zips bundle libtorch/ — no separate download needed.
download_release() {
    if [ -d "${INSTALL_DIR}" ]; then
        ok "${INSTALL_DIR}/ already exists — skipping download."
        return
    fi

    local zip_name="${ASSET_NAME}.zip"
    local download_url="https://github.com/${REPO}/releases/latest/download/${zip_name}"

    info "Downloading ${zip_name} ..."
    curl -fSL -o "${zip_name}" "${download_url}"
    info "Extracting ..."
    unzip -q "${zip_name}"
    mv "${ASSET_NAME}" "${INSTALL_DIR}"
    rm -f "${zip_name}"
    ok "Release extracted to ${INSTALL_DIR}/"
}

choose_model() {
    echo ""
    info "Available models:"
    echo "  1) Qwen3-ASR-0.6B  (recommended — ~1.2 GB download)"
    echo "  2) Qwen3-ASR-1.7B  (~3.5 GB download)"
    echo "  3) Oruk Fourier-2B (English-only, CC BY-NC 4.0 — ~4.1 GB download)"
    echo ""

    local choice
    read -r -p "Select model [1]: " choice </dev/tty
    choice="${choice:-1}"

    case "$choice" in
        1)
            MODEL="Qwen3-ASR-0.6B"
            HF_REPO="Qwen/Qwen3-ASR-0.6B"
            ;;
        2)
            MODEL="Qwen3-ASR-1.7B"
            HF_REPO="Qwen/Qwen3-ASR-1.7B"
            ;;
        3)
            MODEL="oruk-fourier-2b"
            HF_REPO="oruk/oruk-fourier-2b"
            ;;
        *)
            warn "Invalid choice '${choice}', defaulting to 0.6B."
            MODEL="Qwen3-ASR-0.6B"
            HF_REPO="Qwen/Qwen3-ASR-0.6B"
            ;;
    esac

    MODEL_DIR="${INSTALL_DIR}/${MODEL}"
    info "Selected model: ${MODEL}"
}

# ── Download model weights ───────────────────────────────────────────
download_model() {
    if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
        ok "Model ${MODEL} already downloaded — skipping."
        return
    fi

    mkdir -p "${MODEL_DIR}"
    local base_url="https://huggingface.co/${HF_REPO}/resolve/main"

    local files="config.json"
    if [ "$MODEL" = "Qwen3-ASR-0.6B" ]; then
        files="$files model.safetensors"
    elif [ "$MODEL" = "oruk-fourier-2b" ]; then
        files="$files model.safetensors tokenizer.json"
    else
        files="$files model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors"
    fi

    info "Downloading ${MODEL} from HuggingFace (this may take a while) ..."
    for f in $files; do
        if [ -f "${MODEL_DIR}/${f}" ]; then
            ok "${f} already exists — skipping."
        else
            info "  Downloading ${f} ..."
            curl -fSL -o "${MODEL_DIR}/${f}" "${base_url}/${f}"
        fi
    done
    ok "Model downloaded to ${MODEL_DIR}/"
}

# ── Copy pre-built tokenizer ─────────────────────────────────────────
install_tokenizer() {
    if [ -f "${MODEL_DIR}/tokenizer.json" ]; then
        ok "tokenizer.json already exists — skipping."
        return
    fi

    local size
    size=$(echo "$MODEL" | grep -oE '[0-9]+\.[0-9]+B')
    local src="${INSTALL_DIR}/tokenizers/tokenizer-${size}.json"

    if [ ! -f "$src" ]; then
        err "Pre-built tokenizer not found at ${src}"
        exit 1
    fi

    info "Copying pre-built tokenizer ..."
    cp "$src" "${MODEL_DIR}/tokenizer.json"
    ok "Tokenizer installed to ${MODEL_DIR}/tokenizer.json"
}

# ── Download sample audio ────────────────────────────────────────────
download_sample() {
    local dest="${INSTALL_DIR}/sample.wav"

    if [ -f "${dest}" ]; then
        ok "sample.wav already exists — skipping."
        return
    fi

    info "Downloading sample audio file ..."
    curl -fSL -o "${dest}" "${SAMPLE_WAV_URL}"
    ok "Sample saved to ${dest}"
}

# ── Print usage instructions ─────────────────────────────────────────
print_usage() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN} Installation complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""

    echo "Run your first transcription:"
    echo ""
    echo -e "  ${CYAN}cd ${INSTALL_DIR}${NC}"
    echo -e "  ${CYAN}./asr ./${MODEL} sample.wav${NC}"
    echo ""
    echo "Expected output:"
    echo ""
    echo "  Language: English"
    echo "  Text: Thank you for your contribution to the most recent issue of Computer."
    echo ""
    echo "To transcribe your own files:"
    echo ""
    echo -e "  ${CYAN}./asr ./${MODEL} /path/to/audio.wav${NC}"
    echo ""
}

# ── main ─────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║     Qwen3-ASR Installer              ║"
    echo "╚══════════════════════════════════════╝"

    detect_platform
    print_platform
    resolve_asset
    download_release
    choose_model
    download_model
    install_tokenizer
    download_sample
    print_usage
}

main "$@"
