use anyhow::Result;
use std::collections::HashMap;
use tch::{Kind, Tensor};

use crate::config::TextDecoderConfig;
use crate::layers::{RmsNorm, TextDecoderLayer};
use crate::weights::get_weight;

/// KV cache for autoregressive generation.
pub struct KvCache {
    /// Per-layer cache: (key, value) each (batch, num_kv_heads, seq_len, head_dim)
    pub layers: Vec<Option<(Tensor, Tensor)>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(None);
        }
        Self { layers }
    }

    pub fn get(&self, layer: usize) -> Option<&(Tensor, Tensor)> {
        self.layers[layer].as_ref()
    }

    pub fn set(&mut self, layer: usize, cache: (Tensor, Tensor)) {
        self.layers[layer] = Some(cache);
    }

    /// Get the total cached sequence length (from first layer).
    pub fn seq_len(&self) -> i64 {
        self.layers[0]
            .as_ref()
            .map(|(k, _)| k.size()[2])
            .unwrap_or(0)
    }
}

/// Qwen3 Text Decoder model.
///
/// Architecture:
/// - Token embedding (vocab_size -> hidden_size)
/// - N decoder layers with GQA, QK-norm, MRoPE, SwiGLU MLP
/// - Final RMSNorm
/// - LM head (hidden_size -> vocab_size, tied with embedding)
pub struct TextDecoder {
    embed_tokens: Tensor,
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
    lm_head_weight: Tensor,
    config: TextDecoderConfig,
}

impl TextDecoder {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        config: &TextDecoderConfig,
    ) -> Result<Self> {
        let embed_tokens = get_weight(weights, &format!("{}.embed_tokens", prefix), "weight")?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = TextDecoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, &format!("{}.norm", prefix), config.rms_norm_eps)?;

        // LM head weight: tied with embed_tokens if configured
        let lm_head_key = format!(
            "{}",
            prefix.replace(".model", ".lm_head")
        );
        let lm_head_weight = if config.tie_word_embeddings {
            embed_tokens.shallow_clone()
        } else {
            get_weight(weights, &lm_head_key, "weight")?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head_weight,
            config: config.clone(),
        })
    }

    /// Get token embeddings for the given input IDs.
    pub fn embed(&self, input_ids: &Tensor) -> Tensor {
        Tensor::embedding(&self.embed_tokens, input_ids, -1, false, false)
    }

    /// Forward pass through the decoder.
    ///
    /// hidden_states: (batch, seq_len, hidden_size)
    /// cos, sin: (seq_len, head_dim) for MRoPE
    /// kv_cache: mutable KV cache for autoregressive generation
    /// mask: causal attention mask
    ///
    /// Returns: logits of shape (batch, seq_len, vocab_size)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let mut hidden = hidden_states.shallow_clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = kv_cache.get(i);
            let (h, new_cache) = layer.forward(&hidden, cos, sin, cache, mask);
            kv_cache.set(i, new_cache);
            hidden = h;
        }

        let hidden = self.norm.forward(&hidden);

        // LM head: project to vocabulary
        hidden.matmul(&self.lm_head_weight.tr())
    }

    pub fn config(&self) -> &TextDecoderConfig {
        &self.config
    }
}

/// Create a causal attention mask.
/// Returns a mask where future positions are -inf, past positions are 0.
pub fn create_causal_mask(seq_len: i64, past_len: i64, device: tch::Device) -> Tensor {
    let total_len = past_len + seq_len;
    // Create a full mask of -inf
    let mask = Tensor::full(
        [seq_len, total_len],
        f64::NEG_INFINITY,
        (Kind::Float, device),
    );
    // Create a lower-triangular mask
    // For each query position q (0..seq_len), allow attention to positions 0..(past_len + q + 1)
    let mask = mask.triu(past_len + 1);
    // Add batch and head dims: (1, 1, seq_len, total_len)
    mask.unsqueeze(0).unsqueeze(0)
}
