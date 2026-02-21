use anyhow::Result;
use std::path::Path;

pub struct AsrTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl AsrTokenizer {
    /// Load tokenizer from model directory.
    /// Expects either tokenizer.json or vocab.json + merges.txt
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            return Ok(Self { tokenizer });
        }

        // If tokenizer.json doesn't exist, generate it from vocab.json and merges.txt
        anyhow::bail!(
            "tokenizer.json not found in {:?}. \
             Please generate it using: \
             python -c \"from transformers import AutoTokenizer; \
             tok = AutoTokenizer.from_pretrained('{}', trust_remote_code=True); \
             tok.backend_tokenizer.save('{}/tokenizer.json')\"",
            model_dir,
            model_dir.display(),
            model_dir.display()
        );
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[i64]) -> Result<String> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        let text = self
            .tokenizer
            .decode(&u32_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        Ok(text)
    }
}

// Special token IDs for Qwen3-ASR
pub const IM_START_TOKEN_ID: i64 = 151644;
pub const IM_END_TOKEN_ID: i64 = 151645;
pub const ENDOFTEXT_TOKEN_ID: i64 = 151643;
pub const AUDIO_START_TOKEN_ID: i64 = 151669;
pub const AUDIO_END_TOKEN_ID: i64 = 151670;
pub const AUDIO_PAD_TOKEN_ID: i64 = 151676;
pub const ASR_TEXT_TOKEN_ID: i64 = 151704;
