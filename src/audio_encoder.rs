use anyhow::Result;
use std::collections::HashMap;
use tch::{Kind, Tensor};

use crate::config::AudioEncoderConfig;
use crate::layers::{AudioEncoderLayer, Conv2d, LayerNorm, Linear};

/// Qwen3 ASR Audio Encoder (Whisper-style with chunk-based processing).
///
/// Architecture:
/// 1. Split mel into chunks of n_window*2 frames
/// 2. Batch all chunks through 3x Conv2d downsampling (stride 2) -> flatten -> linear
/// 3. Sinusoidal positional embedding (same positions broadcast to all chunks)
/// 4. Extract valid tokens per chunk, concatenate
/// 5. N transformer encoder layers with windowed attention
/// 6. Output projection: LN -> Linear -> GELU -> Linear (d_model -> output_dim)
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
        let num_frames = mel_features.size()[1] as usize;

        // Chunk size = n_window * 2
        let chunk_size = self.config.n_window * 2;

        // Split mel into chunks
        let num_full_chunks = num_frames / chunk_size;
        let tail_frames = num_frames % chunk_size;
        let num_chunks = num_full_chunks + if tail_frames > 0 { 1 } else { 0 };

        // Batch all chunks together (matching Python which processes all chunks as a batch)
        let mut chunk_mels: Vec<Tensor> = Vec::with_capacity(num_chunks);
        let mut chunk_valid_tokens: Vec<usize> = Vec::with_capacity(num_chunks);

        for i in 0..num_full_chunks {
            let start = (i * chunk_size) as i64;
            let chunk_mel = mel_features
                .narrow(1, start, chunk_size as i64)
                .unsqueeze(0); // (1, mel_bins, chunk_size)
            chunk_mels.push(chunk_mel);
            chunk_valid_tokens.push(Self::feat_extract_output_length(chunk_size));
        }

        if tail_frames > 0 {
            // Pad tail chunk to chunk_size with zeros (matching Python's pad_sequence)
            let start = (num_full_chunks * chunk_size) as i64;
            let tail_mel = mel_features.narrow(1, start, tail_frames as i64);
            let pad_frames = chunk_size - tail_frames;
            let padded_mel = Tensor::cat(
                &[
                    &tail_mel,
                    &Tensor::zeros(
                        [mel_features.size()[0], pad_frames as i64],
                        (tch::Kind::Float, mel_features.device()),
                    ),
                ],
                1,
            )
            .unsqueeze(0); // (1, mel_bins, chunk_size)
            chunk_mels.push(padded_mel);
            chunk_valid_tokens.push(Self::feat_extract_output_length(tail_frames));
        }

        // Batch all chunks: (num_chunks, 1, mel_bins, chunk_size)
        let chunk_refs: Vec<&Tensor> = chunk_mels.iter().collect();
        let batched = Tensor::cat(&chunk_refs, 0).unsqueeze(1);

        // Process all chunks through Conv2d stem as a batch
        let x = self.conv2d1.forward(&batched).gelu("none");
        let x = self.conv2d2.forward(&x).gelu("none");
        let x = self.conv2d3.forward(&x).gelu("none");

        // Reshape: (b, channels, freq, time) -> (b, time, channels*freq)
        let (b, c, f, t) = x.size4().unwrap();
        let reshaped = x.permute([0, 3, 1, 2]).contiguous().reshape([b, t, c * f]);
        let conv_out = self.conv_out.forward(&reshaped);

        // Add positional embedding (same for all chunks, matching Python's broadcast)
        let pos_emb = self.positional_embedding.narrow(0, 0, t).unsqueeze(0);
        let conv_out = conv_out + pos_emb;

        // Extract valid tokens per chunk, concatenate into flat sequence
        let mut all_valid: Vec<Tensor> = Vec::new();
        for (i, &valid) in chunk_valid_tokens.iter().enumerate() {
            let chunk_tokens = conv_out.get(i as i64).narrow(0, 0, valid as i64);
            all_valid.push(chunk_tokens);
        }

        // Concatenate: (total_tokens, d_model)
        let valid_refs: Vec<&Tensor> = all_valid.iter().collect();
        let hidden = Tensor::cat(&valid_refs, 0);
        let total_tokens = hidden.size()[0];

        // Add batch dim for transformer: (1, total_tokens, d_model)
        let mut hidden = hidden.unsqueeze(0);

        // Build windowed attention mask
        let mask = self.build_window_mask(total_tokens, &chunk_valid_tokens, mel_features.device());

        // Transformer encoder layers with windowed attention
        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask.as_ref());
        }

        // Output projection: LN -> Linear -> GELU -> Linear
        let hidden = self.ln_post.forward(&hidden);
        let hidden = self.proj1.forward(&hidden).gelu("none");
        let hidden = self.proj2.forward(&hidden);

        // Remove batch dim: (num_tokens, output_dim)
        hidden.squeeze_dim(0)
    }

    /// Build a block-diagonal windowed attention mask.
    ///
    /// Tokens within the same window can attend to each other;
    /// tokens in different windows cannot.
    ///
    /// Returns None if all tokens fit in a single window (no masking needed).
    fn build_window_mask(
        &self,
        total_tokens: i64,
        chunk_token_counts: &[usize],
        device: tch::Device,
    ) -> Option<Tensor> {
        let chunk_size = self.config.n_window * 2;
        // Number of chunks per attention window
        let chunks_per_window = self.config.n_window_infer / chunk_size;

        if chunks_per_window == 0 || chunk_token_counts.len() <= chunks_per_window {
            // All tokens fit in one window, no masking needed
            return None;
        }

        // Group chunks into windows
        let num_windows = (chunk_token_counts.len() + chunks_per_window - 1) / chunks_per_window;

        // Build mask: -inf for positions that cannot attend, 0 for positions that can
        let mask = Tensor::full(
            [1, 1, total_tokens, total_tokens],
            f64::NEG_INFINITY,
            (Kind::Float, device),
        );

        let mut token_offset: i64 = 0;
        for w in 0..num_windows {
            let chunk_start = w * chunks_per_window;
            let chunk_end = std::cmp::min(chunk_start + chunks_per_window, chunk_token_counts.len());

            let window_tokens: i64 = chunk_token_counts[chunk_start..chunk_end]
                .iter()
                .map(|&c| c as i64)
                .sum();

            // Allow full attention within this window
            let _ = mask
                .narrow(2, token_offset, window_tokens)
                .narrow(3, token_offset, window_tokens)
                .fill_(0.0);

            token_offset += window_tokens;
        }

        Some(mask)
    }

    /// Compute output token count for a given number of input frames through 3x Conv2d.
    /// Matches Python's _get_feat_extract_output_lengths.
    fn feat_extract_output_length(input_frames: usize) -> usize {
        let after_conv = |len: usize| -> usize { (len - 1) / 2 + 1 };
        after_conv(after_conv(after_conv(input_frames)))
    }

    /// Get the number of output audio tokens for a given number of mel frames.
    pub fn get_output_length(&self, input_frames: usize) -> usize {
        let chunk_size = self.config.n_window * 2;
        let num_full_chunks = input_frames / chunk_size;
        let tail_frames = input_frames % chunk_size;

        let mut total = num_full_chunks * Self::feat_extract_output_length(chunk_size);
        if tail_frames > 0 {
            total += Self::feat_extract_output_length(tail_frames);
        }
        total
    }
}

/// Create sinusoidal positional embeddings.
/// Uses the standard Transformer sin/cos scheme with geometric timescale progression.
/// Formula: inv_timescales = exp(-i * log(10000) / (half_dim - 1))
fn create_sinusoidal_embedding(max_len: usize, dim: usize, device: tch::Device) -> Tensor {
    let half_dim = dim / 2;
    let log_timescale_increment = (10000.0f64).ln() / (half_dim - 1) as f64;

    let mut embeddings = vec![0.0f32; max_len * dim];

    for pos in 0..max_len {
        for i in 0..half_dim {
            let inv_timescale = (-(i as f64) * log_timescale_increment).exp();
            let angle = pos as f64 * inv_timescale;
            embeddings[pos * dim + i] = angle.sin() as f32;
            embeddings[pos * dim + half_dim + i] = angle.cos() as f32;
        }
    }

    Tensor::from_slice(&embeddings)
        .reshape([max_len as i64, dim as i64])
        .to_device(device)
}
