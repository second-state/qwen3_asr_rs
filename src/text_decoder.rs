use anyhow::Result;
use std::collections::HashMap;
use crate::tensor::{DType, Device, Tensor};

use crate::config::TextDecoderConfig;
use crate::layers::{RmsNorm, TextDecoderLayer};
use crate::weights::get_weight;

/// KV cache for autoregressive generation.
pub struct KvCache {
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

    pub fn seq_len(&self) -> i64 {
        self.layers[0]
            .as_ref()
            .map(|(k, _)| k.size()[2])
            .unwrap_or(0)
    }
}

/// Qwen3 Text Decoder model.
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

    pub fn embed(&self, input_ids: &Tensor) -> Tensor {
        Tensor::embedding(&self.embed_tokens, input_ids)
    }

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
        hidden.matmul(&self.lm_head_weight.tr())
    }

    pub fn config(&self) -> &TextDecoderConfig {
        &self.config
    }
}

/// Create a causal attention mask.
pub fn create_causal_mask(seq_len: i64, past_len: i64, device: Device) -> Tensor {
    let total_len = past_len + seq_len;
    let mask = Tensor::full(
        &[seq_len, total_len],
        f64::NEG_INFINITY,
        DType::Float32,
        device,
    );
    let mask = mask.triu(past_len + 1);
    mask.unsqueeze(0).unsqueeze(0)
}
