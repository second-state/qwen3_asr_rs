# Qwen3 ASR — Voice Transcription

Transcribe speech from audio files to text.

## Binary

- `{baseDir}/scripts/asr` — Speech-to-text transcription.

## Models

- `{baseDir}/scripts/models/Qwen3-ASR-0.6B` — Faster, lighter model (0.6B parameters).
- `{baseDir}/scripts/models/Qwen3-ASR-1.7B` — Higher accuracy model (1.7B parameters).

## Environment Setup

On Linux, set `LD_LIBRARY_PATH` before running:

```shell
export LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH
```

On macOS, set `DYLD_LIBRARY_PATH` before running:

```shell
export DYLD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$DYLD_LIBRARY_PATH
```

## Transcription

Transcribe an audio file to text.

```shell
# Linux
LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  {baseDir}/scripts/asr \
  {baseDir}/scripts/models/Qwen3-ASR-0.6B \
  <audio_file>

# macOS
DYLD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$DYLD_LIBRARY_PATH \
  {baseDir}/scripts/asr \
  {baseDir}/scripts/models/Qwen3-ASR-0.6B \
  <audio_file>
```

### Parameters

| Parameter  | Required | Description                                        |
|------------|----------|----------------------------------------------------|
| model_path | Yes      | Path to the model directory (0.6B or 1.7B)         |
| audio_file | Yes      | Path to the audio file (any FFmpeg-supported format)|

### Output

Prints the transcribed text to standard output.

### Example

```shell
# Linux
LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  {baseDir}/scripts/asr \
  {baseDir}/scripts/models/Qwen3-ASR-0.6B \
  recording.wav

# macOS
DYLD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$DYLD_LIBRARY_PATH \
  {baseDir}/scripts/asr \
  {baseDir}/scripts/models/Qwen3-ASR-0.6B \
  recording.wav
```

## Model Selection

- Use **0.6B** for fast transcription where speed matters more than accuracy.
- Use **1.7B** for higher accuracy, especially on difficult audio, accents, or low-quality recordings.

## Supported Audio Formats

Any format supported by FFmpeg: WAV, MP3, M4A, FLAC, OGG, and more. Audio is automatically resampled to 16 kHz mono internally.

## Workflow

### 1. Identify the Audio File

Get the path to the audio file the user wants to transcribe.

### 2. Choose a Model

- Default to **0.6B** unless the user asks for higher accuracy or the 0.6B result is unsatisfactory.
- Switch to **1.7B** when the user requests it, or when the audio is noisy, heavily accented, or the 0.6B output looks wrong.

### 3. Run the Command

Run the `asr` binary with the library path set. Use the full paths to the binary and model directory.

```shell
# Linux
LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  {baseDir}/scripts/asr \
  {baseDir}/scripts/models/Qwen3-ASR-0.6B \
  /path/to/audio.mp3

# macOS
DYLD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$DYLD_LIBRARY_PATH \
  {baseDir}/scripts/asr \
  {baseDir}/scripts/models/Qwen3-ASR-0.6B \
  /path/to/audio.mp3
```

### 4. Return the Transcription

The transcribed text is printed to stdout. Return it to the user.
