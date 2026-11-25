use serde::Deserialize;
use std::collections::HashMap;

/// Top-level model configuration (loaded from config.json).
///
/// Supports both the legacy nested `thinker_config` layout (e.g., `Qwen/Qwen3-ASR-1.7B`)
/// and the native HuggingFace `transformers>=5.21` flat layout (e.g., `oruk/oruk-fourier-2b`,
/// `Qwen/Qwen3-ASR-1.7B-hf`).
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "RawAsrConfig")]
pub struct AsrConfig {
    pub thinker_config: ThinkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RawAsrConfig {
    Nested {
        thinker_config: ThinkerConfig,
    },
    Flat(FlatAsrConfig),
}

#[derive(Debug, Clone, Deserialize)]
struct FlatAsrConfig {
    audio_config: AudioEncoderConfig,
    text_config: TextDecoderConfig,
    #[serde(default = "default_audio_start_token_id")]
    audio_start_token_id: i64,
    #[serde(default = "default_audio_end_token_id")]
    audio_end_token_id: i64,
    #[serde(default = "default_audio_token_id")]
    audio_token_id: i64,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

impl From<RawAsrConfig> for AsrConfig {
    fn from(raw: RawAsrConfig) -> Self {
        match raw {
            RawAsrConfig::Nested { thinker_config } => Self { thinker_config },
            RawAsrConfig::Flat(mut flat) => {
                if let Some(tie) = flat.tie_word_embeddings {
                    if !flat.text_config.tie_word_embeddings_explicit {
                        flat.text_config.tie_word_embeddings = tie;
                    }
                }
                Self {
                    thinker_config: ThinkerConfig {
                        audio_config: flat.audio_config,
                        text_config: flat.text_config,
                        audio_start_token_id: flat.audio_start_token_id,
                        audio_end_token_id: flat.audio_end_token_id,
                        audio_token_id: flat.audio_token_id,
                    },
                }
            }
        }
    }
}

/// Safetensors key prefixes for a Qwen3-ASR checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightPrefixes {
    pub audio_tower: &'static str,
    pub proj1: &'static str,
    pub proj2: &'static str,
    pub text_model: &'static str,
    pub lm_head: &'static str,
}

pub const LEGACY_WEIGHT_PREFIXES: WeightPrefixes = WeightPrefixes {
    audio_tower: "thinker.audio_tower",
    proj1: "thinker.audio_tower.proj1",
    proj2: "thinker.audio_tower.proj2",
    text_model: "thinker.model",
    lm_head: "thinker.lm_head",
};

pub const HF_NATIVE_WEIGHT_PREFIXES: WeightPrefixes = WeightPrefixes {
    audio_tower: "model.audio_tower",
    proj1: "model.multi_modal_projector.linear_1",
    proj2: "model.multi_modal_projector.linear_2",
    text_model: "model.language_model",
    lm_head: "lm_head",
};

/// Detect whether the loaded safetensors map uses legacy `thinker.*` keys
/// or native HuggingFace `transformers>=5.21` (`model.audio_tower.*`,
/// `model.multi_modal_projector.linear_{1,2}.*`, `model.language_model.*`) keys.
pub fn detect_weight_prefixes<V>(weights: &HashMap<String, V>) -> WeightPrefixes {
    if weights.contains_key("thinker.audio_tower.conv2d1.weight") {
        LEGACY_WEIGHT_PREFIXES
    } else if weights.contains_key("model.audio_tower.conv2d1.weight") {
        HF_NATIVE_WEIGHT_PREFIXES
    } else {
        LEGACY_WEIGHT_PREFIXES
    }
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
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,
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
fn default_n_window_infer() -> usize { 800 }
fn default_conv_chunksize() -> usize { 500 }
fn default_downsample_hidden_size() -> i64 { 480 }
fn default_output_dim() -> i64 { 1024 }

/// Text decoder configuration (Qwen3-based)
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "RawTextDecoderConfig")]
pub struct TextDecoderConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub tie_word_embeddings: bool,
    pub(crate) tie_word_embeddings_explicit: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RawTextDecoderConfig {
    #[serde(default = "default_vocab_size")]
    vocab_size: i64,
    #[serde(default = "default_hidden_size")]
    hidden_size: i64,
    #[serde(default = "default_intermediate_size")]
    intermediate_size: i64,
    #[serde(default = "default_num_hidden_layers")]
    num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    #[serde(default)]
    rope_theta: Option<f64>,
}

impl From<RawTextDecoderConfig> for TextDecoderConfig {
    fn from(raw: RawTextDecoderConfig) -> Self {
        let rope_theta = raw
            .rope_theta
            .or_else(|| raw.rope_parameters.as_ref().and_then(|rp| rp.rope_theta))
            .unwrap_or_else(default_rope_theta);
        let tie_word_embeddings_explicit = raw.tie_word_embeddings.is_some();
        let tie_word_embeddings = raw
            .tie_word_embeddings
            .unwrap_or_else(default_tie_word_embeddings);
        Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta,
            rope_scaling: raw.rope_scaling,
            tie_word_embeddings,
            tie_word_embeddings_explicit,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_legacy_thinker_config() {
        let json = r#"{
            "thinker_config": {
                "audio_config": {
                    "d_model": 1024,
                    "encoder_layers": 24,
                    "encoder_attention_heads": 16,
                    "encoder_ffn_dim": 4096,
                    "num_mel_bins": 128,
                    "max_source_positions": 1500,
                    "n_window": 50,
                    "n_window_infer": 800,
                    "conv_chunksize": 500,
                    "downsample_hidden_size": 480,
                    "output_dim": 2048
                },
                "text_config": {
                    "vocab_size": 151936,
                    "hidden_size": 2048,
                    "intermediate_size": 6144,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 1000000.0,
                    "tie_word_embeddings": true
                },
                "audio_start_token_id": 151669,
                "audio_end_token_id": 151670,
                "audio_token_id": 151676
            }
        }"#;

        let cfg: AsrConfig = serde_json::from_str(json).expect("legacy config should parse");
        assert_eq!(cfg.thinker_config.audio_config.d_model, 1024);
        assert_eq!(cfg.thinker_config.audio_config.encoder_layers, 24);
        assert_eq!(cfg.thinker_config.text_config.hidden_size, 2048);
        assert_eq!(cfg.thinker_config.text_config.rope_theta, 1_000_000.0);
        assert!(cfg.thinker_config.text_config.tie_word_embeddings);
        assert_eq!(cfg.thinker_config.audio_token_id, 151676);
    }

    #[test]
    fn parses_native_hf_fourier_2b_config() {
        let json = r#"{
            "architectures": ["Qwen3ASRForConditionalGeneration"],
            "audio_config": {
                "conv_chunksize": 500,
                "d_model": 1024,
                "downsample_hidden_size": 480,
                "encoder_attention_heads": 16,
                "encoder_ffn_dim": 4096,
                "encoder_layers": 24,
                "max_position_embeddings": 13,
                "n_window": 50,
                "n_window_infer": 800,
                "num_mel_bins": 128,
                "output_dim": 2048
            },
            "audio_token_id": 151676,
            "model_type": "qwen3_asr",
            "text_config": {
                "head_dim": 128,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "rope_theta": 1000000,
                    "rope_type": "default"
                },
                "tie_word_embeddings": true,
                "vocab_size": 151936
            },
            "tie_word_embeddings": true
        }"#;

        let cfg: AsrConfig = serde_json::from_str(json).expect("native HF config should parse");
        assert_eq!(cfg.thinker_config.audio_config.d_model, 1024);
        assert_eq!(cfg.thinker_config.audio_config.encoder_layers, 24);
        assert_eq!(cfg.thinker_config.audio_config.output_dim, 2048);
        assert_eq!(cfg.thinker_config.audio_config.max_source_positions, 1500);
        assert_eq!(cfg.thinker_config.text_config.hidden_size, 2048);
        assert_eq!(cfg.thinker_config.text_config.intermediate_size, 6144);
        assert_eq!(cfg.thinker_config.text_config.num_hidden_layers, 28);
        assert_eq!(cfg.thinker_config.text_config.rope_theta, 1_000_000.0);
        assert!(cfg.thinker_config.text_config.tie_word_embeddings);
        assert_eq!(cfg.thinker_config.audio_start_token_id, 151669);
        assert_eq!(cfg.thinker_config.audio_end_token_id, 151670);
        assert_eq!(cfg.thinker_config.audio_token_id, 151676);
    }

    #[test]
    fn extracts_custom_rope_parameters_and_top_level_tie_flag() {
        let json = r#"{
            "audio_config": { "d_model": 1024 },
            "text_config": {
                "hidden_size": 2048,
                "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" }
            },
            "tie_word_embeddings": false
        }"#;

        let cfg: AsrConfig = serde_json::from_str(json).expect("flat config should parse");
        assert_eq!(cfg.thinker_config.text_config.rope_theta, 500_000.0);
        assert!(!cfg.thinker_config.text_config.tie_word_embeddings);
    }

    #[test]
    fn detects_legacy_and_native_hf_weight_prefixes() {
        let mut legacy = HashMap::new();
        legacy.insert("thinker.audio_tower.conv2d1.weight".to_string(), ());
        assert_eq!(detect_weight_prefixes(&legacy), LEGACY_WEIGHT_PREFIXES);

        let mut native = HashMap::new();
        native.insert("model.audio_tower.conv2d1.weight".to_string(), ());
        assert_eq!(detect_weight_prefixes(&native), HF_NATIVE_WEIGHT_PREFIXES);
    }
}
