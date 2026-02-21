#!/bin/bash
# Bootstrap script for Qwen3 ASR skill
# Downloads platform-specific binary, libtorch, and models

set -e

REPO="second-state/qwen3_asr_rs"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SKILL_DIR}/scripts"
MODELS_DIR="${SCRIPTS_DIR}/models"

detect_platform() {
    local os arch

    case "$(uname -s)" in
    Linux*) os="linux" ;;
    Darwin*) os="darwin" ;;
    *)
        echo "Error: Unsupported operating system: $(uname -s)" >&2
        exit 1
        ;;
    esac

    case "$(uname -m)" in
    x86_64 | amd64) arch="x86_64" ;;
    aarch64 | arm64) arch="aarch64" ;;
    *)
        echo "Error: Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
    esac

    echo "${os}-${arch}"
}

get_asset_name() {
    local platform="$1"

    case "$platform" in
    linux-x86_64)
        echo "asr-linux-x86_64"
        ;;
    linux-aarch64)
        echo "asr-linux-aarch64"
        ;;
    darwin-aarch64)
        echo "asr-macos-aarch64"
        ;;
    *)
        echo "Error: Unsupported platform: ${platform}" >&2
        exit 1
        ;;
    esac
}

download_binary() {
    local asset_name="$1"

    echo "=== Downloading binary (${asset_name}) ===" >&2

    mkdir -p "${SCRIPTS_DIR}"

    # Get download URL from latest release
    local api_url="https://api.github.com/repos/${REPO}/releases/latest"
    local download_url
    download_url=$(curl -sL "$api_url" | grep -o "https://github.com/${REPO}/releases/download/[^\"]*/${asset_name}" | head -1)

    if [ -z "$download_url" ]; then
        echo "Error: Could not find release asset ${asset_name}" >&2
        echo "Check https://github.com/${REPO}/releases for available downloads." >&2
        exit 1
    fi

    echo "Fetching from: ${download_url}" >&2
    curl -sL -o "${SCRIPTS_DIR}/asr" "$download_url"
    chmod +x "${SCRIPTS_DIR}/asr"

    echo "Binary installed to ${SCRIPTS_DIR}/asr" >&2
}

download_libtorch() {
    local platform="$1"

    echo "=== Downloading libtorch ===" >&2

    local libtorch_url=""
    local archive_name=""

    case "$platform" in
    linux-x86_64)
        libtorch_url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip"
        archive_name="libtorch.zip"
        ;;
    linux-aarch64)
        libtorch_url="https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz"
        archive_name="libtorch.tar.gz"
        ;;
    darwin-aarch64)
        libtorch_url="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip"
        archive_name="libtorch.zip"
        ;;
    *)
        echo "Error: No libtorch URL for platform ${platform}" >&2
        exit 1
        ;;
    esac

    local temp_dir
    temp_dir=$(mktemp -d)

    echo "Fetching from: ${libtorch_url}" >&2
    curl -sL -o "${temp_dir}/${archive_name}" "$libtorch_url"

    echo "Extracting libtorch..." >&2
    if [[ "$archive_name" == *.zip ]]; then
        unzip -q "${temp_dir}/${archive_name}" -d "${temp_dir}"
    else
        tar xzf "${temp_dir}/${archive_name}" -C "${temp_dir}"
    fi

    rm -rf "${SCRIPTS_DIR}/libtorch"
    mv "${temp_dir}/libtorch" "${SCRIPTS_DIR}/libtorch"

    rm -rf "$temp_dir"
    echo "libtorch installed to ${SCRIPTS_DIR}/libtorch" >&2
}

download_models() {
    echo "=== Downloading models ===" >&2

    mkdir -p "${MODELS_DIR}"

    # Ensure huggingface_hub and transformers are available
    if ! command -v huggingface-cli &>/dev/null; then
        echo "Installing huggingface_hub and transformers..." >&2
        pip install -q huggingface_hub transformers
    fi

    for model in Qwen3-ASR-0.6B Qwen3-ASR-1.7B; do
        local model_dir="${MODELS_DIR}/${model}"
        if [ ! -d "$model_dir" ] || [ -z "$(ls -A "$model_dir" 2>/dev/null)" ]; then
            echo "Downloading ${model}..." >&2
            huggingface-cli download "Qwen/${model}" --local-dir "$model_dir"
        else
            echo "${model} already downloaded, skipping." >&2
        fi
    done

    # Generate tokenizer.json files
    echo "Generating tokenizer.json files..." >&2
    python3 -c "
from transformers import AutoTokenizer
for model in ['Qwen3-ASR-0.6B', 'Qwen3-ASR-1.7B']:
    path = '${MODELS_DIR}/' + model
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.backend_tokenizer.save(path + '/tokenizer.json')
    print(f'Saved {path}/tokenizer.json')
"

    echo "Models installed to ${MODELS_DIR}" >&2
}

main() {
    local platform
    platform=$(detect_platform)
    echo "Detected platform: ${platform}" >&2

    local asset_name
    asset_name=$(get_asset_name "$platform")
    echo "Asset: ${asset_name}" >&2

    download_binary "$asset_name"
    download_libtorch "$platform"
    download_models

    echo "" >&2
    echo "=== Installation complete ===" >&2
    echo "Installed files:" >&2
    ls -1 "${SCRIPTS_DIR}" | grep -v '^\.' >&2
}

main "$@"
