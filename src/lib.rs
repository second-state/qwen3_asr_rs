// Ensure exactly one backend is selected
#[cfg(all(feature = "tch-backend", feature = "mlx"))]
compile_error!("Features 'tch-backend' and 'mlx' are mutually exclusive");

#[cfg(not(any(feature = "tch-backend", feature = "mlx")))]
compile_error!("Either 'tch-backend' or 'mlx' feature must be enabled");

pub mod tensor;
#[cfg(feature = "mlx")]
pub mod backend;

pub mod audio;
pub mod audio_encoder;
pub mod config;
pub mod error;
pub mod inference;
pub mod layers;
pub mod mel;
pub mod text_decoder;
pub mod tokenizer;
pub mod weights;
