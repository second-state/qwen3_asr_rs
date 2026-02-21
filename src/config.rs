use serde::Deserialize;

/// Top-level model configuration (loaded from config.json)
#[derive(Debug, Clone, Deserialize)]
pub struct AsrConfig {
    pub thinker_config: ThinkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfig {
    pub audio_config: AudioEncoderConfig,
    pub text_config: TextDecoderConfig,
    #[serde(default = "default_audio_start_token_id")]
    pub audio_start_token_id: i64,
    #[serde(default = "default_audio_end_token_id")]
    pub audio_end_token_id: i64,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: i64,
}

fn default_audio_start_token_id() -> i64 { 151669 }
fn default_audio_end_token_id() -> i64 { 151670 }
fn default_audio_token_id() -> i64 { 151676 }

/// Audio encoder configuration (Whisper-style)
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    #[serde(default = "default_d_model")]
    pub d_model: i64,
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: i64,
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    #[serde(default = "default_downsample_hidden_size")]
    pub downsample_hidden_size: i64,
    #[serde(default = "default_output_dim")]
    pub output_dim: i64,
}

fn default_d_model() -> i64 { 896 }
fn default_encoder_layers() -> usize { 18 }
fn default_encoder_attention_heads() -> usize { 14 }
fn default_encoder_ffn_dim() -> i64 { 3584 }
fn default_num_mel_bins() -> usize { 128 }
fn default_max_source_positions() -> usize { 1500 }
fn default_n_window() -> usize { 50 }
fn default_downsample_hidden_size() -> i64 { 480 }
fn default_output_dim() -> i64 { 1024 }

/// Text decoder configuration (Qwen3-based)
#[derive(Debug, Clone, Deserialize)]
pub struct TextDecoderConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i64,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i64,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i64,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

fn default_vocab_size() -> i64 { 151936 }
fn default_hidden_size() -> i64 { 1024 }
fn default_intermediate_size() -> i64 { 3072 }
fn default_num_hidden_layers() -> usize { 28 }
fn default_num_attention_heads() -> usize { 16 }
fn default_num_key_value_heads() -> usize { 8 }
fn default_head_dim() -> usize { 128 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_tie_word_embeddings() -> bool { true }

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub rope_type: String,
    #[serde(default = "default_mrope_section")]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub interleaved: bool,
    #[serde(default)]
    pub mrope_interleaved: bool,
}

fn default_mrope_section() -> Vec<usize> { vec![24, 20, 20] }

impl AsrConfig {
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
}

impl TextDecoderConfig {
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_section.clone())
            .unwrap_or_else(default_mrope_section)
    }

    pub fn mrope_interleaved(&self) -> bool {
        self.rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_interleaved || rs.interleaved)
            .unwrap_or(false)
    }
}
