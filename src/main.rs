use anyhow::{Context, Result};
use std::path::Path;
use tch::Device;

use qwen3_asr::inference::AsrInference;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Qwen3 ASR - Automatic Speech Recognition");
        eprintln!();
        eprintln!("Usage: asr <model_path> <audio_file> [language]");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  model_path   Path to the Qwen3-ASR model directory");
        eprintln!("  audio_file   Path to the input audio file (any format supported by ffmpeg)");
        eprintln!("  language     Optional: force language (e.g., chinese, english, japanese)");
        eprintln!();
        eprintln!("The audio file will be automatically converted to mono 16kHz f32 for the model.");
        eprintln!();
        eprintln!("Environment variables:");
        eprintln!("  LIBTORCH     Path to libtorch installation");
        eprintln!("  RUST_LOG     Set logging level (e.g., info, debug, trace)");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let audio_file = &args[2];
    let language = args.get(3).map(|s| s.as_str());

    // Verify paths exist
    let model_dir = Path::new(model_path);
    if !model_dir.exists() {
        anyhow::bail!("Model directory not found: {}", model_path);
    }
    if !Path::new(audio_file).exists() {
        anyhow::bail!("Audio file not found: {}", audio_file);
    }

    // Select device
    let device = if tch::Cuda::is_available() {
        tracing::info!("Using CUDA device");
        Device::Cuda(0)
    } else {
        tracing::info!("Using CPU device");
        Device::Cpu
    };

    // Load model
    let model = AsrInference::load(model_dir, device).context("Failed to load model")?;

    // Run transcription
    tracing::info!("Transcribing: {}", audio_file);
    let result = model
        .transcribe(audio_file, language)
        .context("Transcription failed")?;

    // Output result
    println!("Language: {}", result.language);
    println!("Text: {}", result.text);

    Ok(())
}
