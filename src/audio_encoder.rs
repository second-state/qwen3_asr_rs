use anyhow::Result;
use std::collections::HashMap;
use tch::Tensor;

use crate::config::AudioEncoderConfig;
use crate::layers::{AudioEncoderLayer, Conv2d, LayerNorm, Linear};

/// Qwen3 ASR Audio Encoder (Whisper-style).
///
/// Architecture:
/// 1. 3x Conv2d downsampling (stride 2 each, total 8x time reduction)
/// 2. Linear projection from flattened conv output to d_model
/// 3. Sinusoidal positional embedding
/// 4. N transformer encoder layers (bidirectional attention)
/// 5. Output projection: LN -> Linear -> GELU -> Linear (d_model -> output_dim)
pub struct AudioEncoder {
    // Convolutional downsampling
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,

    // Positional embedding (sinusoidal, precomputed)
    positional_embedding: Tensor,

    // Transformer encoder layers
    layers: Vec<AudioEncoderLayer>,

    // Output projection
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,

    config: AudioEncoderConfig,
}

impl AudioEncoder {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        config: &AudioEncoderConfig,
        device: tch::Device,
    ) -> Result<Self> {
        let conv2d1 = Conv2d::load(weights, &format!("{}.conv2d1", prefix), [2, 2], [1, 1])?;
        let conv2d2 = Conv2d::load(weights, &format!("{}.conv2d2", prefix), [2, 2], [1, 1])?;
        let conv2d3 = Conv2d::load(weights, &format!("{}.conv2d3", prefix), [2, 2], [1, 1])?;
        let conv_out = Linear::load(weights, &format!("{}.conv_out", prefix))?;

        let mut layers = Vec::new();
        for i in 0..config.encoder_layers {
            let layer = AudioEncoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config.encoder_attention_heads,
                config.d_model as usize,
            )?;
            layers.push(layer);
        }

        let ln_post = LayerNorm::load(weights, &format!("{}.ln_post", prefix), 1e-5)?;
        let proj1 = Linear::load(weights, &format!("{}.proj1", prefix))?;
        let proj2 = Linear::load(weights, &format!("{}.proj2", prefix))?;

        // Create sinusoidal positional embedding
        let positional_embedding = create_sinusoidal_embedding(
            config.max_source_positions,
            config.d_model as usize,
            device,
        );

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            positional_embedding,
            layers,
            ln_post,
            proj1,
            proj2,
            config: config.clone(),
        })
    }

    /// Encode mel spectrogram features into continuous audio embeddings.
    ///
    /// mel_features: (num_mel_bins, num_frames) tensor
    /// Returns: (num_audio_tokens, output_dim) tensor
    pub fn forward(&self, mel_features: &Tensor) -> Tensor {
        let _num_frames = mel_features.size()[1];

        // Add batch and channel dims: (1, 1, num_mel_bins, num_frames)
        let x = mel_features.unsqueeze(0).unsqueeze(0);

        // 3x Conv2d downsampling with GELU
        let x = self.conv2d1.forward(&x).gelu("none");
        let x = self.conv2d2.forward(&x).gelu("none");
        let x = self.conv2d3.forward(&x).gelu("none");

        // x shape: (1, channels, freq_reduced, time_reduced)
        let (bsz, channels, freq, time) = x.size4().unwrap();

        // Reshape: flatten channels and freq, keep time as sequence
        // (1, channels, freq, time) -> (1, time, channels * freq)
        let x = x.permute([0, 3, 1, 2]).reshape([bsz, time, channels * freq]);

        // Linear projection to d_model
        let x = self.conv_out.forward(&x);

        // Add positional embedding (truncate or pad as needed)
        let pos_len = std::cmp::min(time as usize, self.config.max_source_positions);
        let pos_emb = self.positional_embedding.narrow(0, 0, pos_len as i64);
        let x = &x.narrow(1, 0, pos_len as i64) + &pos_emb.unsqueeze(0);

        // Transformer encoder layers
        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, None);
        }

        // Output projection: LN -> Linear -> GELU -> Linear
        let hidden = self.ln_post.forward(&hidden);
        let hidden = self.proj1.forward(&hidden).gelu("none");
        let hidden = self.proj2.forward(&hidden);

        // Remove batch dim: (num_tokens, output_dim)
        hidden.squeeze_dim(0)
    }

    /// Get the number of output audio tokens for a given number of mel frames.
    pub fn get_output_length(&self, input_frames: usize) -> usize {
        // Each Conv2d with stride 2: output = (input - 1) / 2 + 1
        let after_conv = |len: usize| -> usize { (len - 1) / 2 + 1 };
        let time = after_conv(after_conv(after_conv(input_frames)));
        std::cmp::min(time, self.config.max_source_positions)
    }
}

/// Create sinusoidal positional embeddings.
/// Uses the standard Transformer sin/cos scheme with geometric timescale progression.
fn create_sinusoidal_embedding(max_len: usize, dim: usize, device: tch::Device) -> Tensor {
    let half_dim = dim / 2;
    let max_timescale: f64 = 10000.0;

    let mut embeddings = vec![0.0f32; max_len * dim];

    for pos in 0..max_len {
        for i in 0..half_dim {
            let timescale = max_timescale.powf(i as f64 / half_dim as f64);
            let angle = pos as f64 / timescale;
            embeddings[pos * dim + i] = angle.sin() as f32;
            embeddings[pos * dim + half_dim + i] = angle.cos() as f32;
        }
    }

    Tensor::from_slice(&embeddings)
        .reshape([max_len as i64, dim as i64])
        .to_device(device)
}
