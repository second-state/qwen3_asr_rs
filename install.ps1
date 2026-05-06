# install.ps1 — One-step installer for Qwen3-ASR Rust CLI (Windows)
# Downloads the release binary (with bundled libtorch), model weights, and a sample audio file.

$ErrorActionPreference = "Stop"

$REPO = "second-state/qwen3_asr_rs"
$INSTALL_DIR = "qwen3_asr_rs"
$SAMPLE_WAV_URL = "https://github.com/${REPO}/raw/main/test_audio/sample1.wav"

function Info($msg)  { Write-Host "[info]  $msg" -ForegroundColor Cyan }
function Ok($msg)    { Write-Host "[ok]    $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "[warn]  $msg" -ForegroundColor Yellow }
function Err($msg)   { Write-Host "[error] $msg" -ForegroundColor Red; exit 1 }

# ── 1. Detect platform ──────────────────────────────────────────────
$OS = "windows"
$ARCH = if ([Environment]::Is64BitOperatingSystem) { "x86_64" } else { "unsupported" }

if ($ARCH -eq "unsupported") {
    Err "Only 64-bit Windows is supported."
}

Info "System detection"
Write-Host "  OS:           ${OS}"
Write-Host "  CPU:          ${ARCH}"
Write-Host ""

# ── 2. Map platform → release asset ─────────────────────────────────
$ASSET_NAME = "asr-windows-x86_64"

# ── 3. Download & extract release ────────────────────────────────────
if (Test-Path $INSTALL_DIR) {
    Ok "${INSTALL_DIR}\ already exists — skipping download."
} else {
    $zipName = "${ASSET_NAME}.zip"
    $downloadUrl = "https://github.com/${REPO}/releases/latest/download/${zipName}"

    Info "Downloading ${zipName} ..."
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $zipName -UseBasicParsing
    } catch {
        Err "Failed to download release. Windows builds may not be available yet."
    }

    Info "Extracting ..."
    Expand-Archive -Path $zipName -DestinationPath "." -Force
    Rename-Item -Path $ASSET_NAME -NewName $INSTALL_DIR -Force
    Remove-Item -Path $zipName -Force
    Ok "Release extracted to ${INSTALL_DIR}\"
}

# ── 4. Choose model ─────────────────────────────────────────────────
Write-Host ""
Info "Available models:"
Write-Host "  1) Qwen3-ASR-0.6B  (recommended — ~1.2 GB download)"
Write-Host "  2) Qwen3-ASR-1.7B  (~3.5 GB download)"
Write-Host ""

$choice = Read-Host "Select model [1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

switch ($choice) {
    "1" { $MODEL = "Qwen3-ASR-0.6B" }
    "2" { $MODEL = "Qwen3-ASR-1.7B" }
    default {
        Warn "Invalid choice '${choice}', defaulting to 0.6B."
        $MODEL = "Qwen3-ASR-0.6B"
    }
}

$MODEL_DIR = "${INSTALL_DIR}\${MODEL}"
Info "Selected model: ${MODEL}"

# ── 5. Download model weights ────────────────────────────────────────
if ((Test-Path $MODEL_DIR) -and (Test-Path "${MODEL_DIR}\config.json")) {
    Ok "Model ${MODEL} already downloaded — skipping."
} else {
    New-Item -ItemType Directory -Path $MODEL_DIR -Force | Out-Null
    $baseUrl = "https://huggingface.co/Qwen/${MODEL}/resolve/main"

    $files = @("config.json")
    if ($MODEL -eq "Qwen3-ASR-0.6B") {
        $files += "model.safetensors"
    } else {
        $files += @("model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    }

    Info "Downloading ${MODEL} from HuggingFace (this may take a while) ..."
    foreach ($f in $files) {
        $dest = "${MODEL_DIR}\${f}"
        if (Test-Path $dest) {
            Ok "${f} already exists — skipping."
        } else {
            Info "  Downloading ${f} ..."
            $url = "${baseUrl}/${f}"
            try {
                Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
            } catch {
                Err "Failed to download ${f} from ${url}"
            }
        }
    }
    Ok "Model downloaded to ${MODEL_DIR}\"
}

# ── 6. Install tokenizer ──────────────────────────────────────────
$tokenizerDest = "${MODEL_DIR}\tokenizer.json"
if (Test-Path $tokenizerDest) {
    Ok "tokenizer.json already exists — skipping."
} else {
    $size = if ($MODEL -eq "Qwen3-ASR-0.6B") { "0.6B" } else { "1.7B" }
    $src = "${INSTALL_DIR}\tokenizers\tokenizer-${size}.json"

    if (-not (Test-Path $src)) {
        Err "Pre-built tokenizer not found at ${src}"
    }

    Info "Copying pre-built tokenizer ..."
    Copy-Item -Path $src -Destination $tokenizerDest -Force
    Ok "Tokenizer installed to ${tokenizerDest}"
}

# ── 7. Download sample audio ────────────────────────────────────────
$sampleDest = "${INSTALL_DIR}\sample.wav"
if (Test-Path $sampleDest) {
    Ok "sample.wav already exists — skipping."
} else {
    Info "Downloading sample audio file ..."
    try {
        Invoke-WebRequest -Uri $SAMPLE_WAV_URL -OutFile $sampleDest -UseBasicParsing
    } catch {
        Warn "Failed to download sample audio. You can use your own WAV files."
    }
    if (Test-Path $sampleDest) {
        Ok "Sample saved to ${sampleDest}"
    }
}

# ── 8. Print usage instructions ───────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " Installation complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Run your first transcription:"
Write-Host ""
Write-Host "  cd ${INSTALL_DIR}" -ForegroundColor Cyan
Write-Host "  .\asr .\${MODEL} sample.wav" -ForegroundColor Cyan
Write-Host ""
Write-Host "Expected output:"
Write-Host ""
Write-Host "  Language: English"
Write-Host "  Text: Thank you for your contribution to the most recent issue of Computer."
Write-Host ""
Write-Host "To transcribe your own files:"
Write-Host ""
Write-Host "  .\asr .\${MODEL} C:\path\to\audio.wav" -ForegroundColor Cyan
Write-Host ""
