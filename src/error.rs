use thiserror::Error;

#[derive(Error, Debug)]
pub enum AsrError {
    #[error("Audio error: {0}")]
    Audio(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Weight loading error: {0}")]
    Weights(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "tch-backend")]
    #[error("Torch error: {0}")]
    Torch(#[from] tch::TchError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
