use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use qwen3_asr_rs::inference::AsrInference;
use qwen3_asr_rs::tensor::Device;

#[derive(Parser)]
#[command(name = "asr", about = "Qwen3 ASR - Automatic Speech Recognition")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transcribe an audio file
    Transcribe {
        /// Path to the Qwen3-ASR model directory
        model_path: PathBuf,

        /// Path to the input audio file
        audio_file: PathBuf,

        /// Optional: force language (e.g., chinese, english, japanese)
        language: Option<String>,
    },
    /// Launch the GPUI-based voice input GUI
    Gui,
}

fn select_device() -> Device {
    #[cfg(feature = "tch-backend")]
    {
        if tch::Cuda::is_available() {
            tracing::info!("Using CUDA device");
            return Device::Gpu(0);
        }
        tracing::info!("Using CPU device");
        Device::Cpu
    }
    #[cfg(feature = "mlx")]
    {
        qwen3_asr_rs::backend::mlx::stream::init_mlx(true);
        tracing::info!("Using MLX Metal GPU");
        Device::Gpu(0)
    }
    #[cfg(not(any(feature = "tch-backend", feature = "mlx")))]
    Device::Cpu
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Transcribe {
            model_path,
            audio_file,
            language,
        } => {
            if !model_path.exists() {
                anyhow::bail!("Model directory not found: {}", model_path.display());
            }
            if !audio_file.exists() {
                anyhow::bail!("Audio file not found: {}", audio_file.display());
            }

            let device = select_device();
            let model = AsrInference::load(&model_path, device)
                .context("Failed to load model")?;

            tracing::info!("Transcribing: {}", audio_file.display());
            let path_str = audio_file.to_string_lossy();
            let result = model
                .transcribe(&path_str, language.as_deref())
                .context("Transcription failed")?;

            println!("Language: {}", result.language);
            println!("Text: {}", result.text);
        }
        Commands::Gui => {
            qwen3_asr_rs::gui::run_gui();
        }
    }

    Ok(())
}
