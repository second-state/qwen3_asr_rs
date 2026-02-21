use std::collections::HashMap;
use tch::{Kind, Tensor};
use anyhow::Result;
use crate::weights::{get_weight, get_weight_opt};

// ============================================================================
// LayerNorm (with bias, used in audio encoder)
// ============================================================================

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNorm {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, prefix, "weight")?,
            bias: get_weight(weights, prefix, "bias")?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
        let ndim = x.dim();
        x.layer_norm(&[x.size()[ndim - 1]], Some(&self.weight), Some(&self.bias), self.eps, true)
    }
}

// ============================================================================
// RMSNorm (used in text decoder)
// ============================================================================

pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNorm {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, prefix, "weight")?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let dtype = x.kind();
        let x = x.to_kind(Kind::Float);
        let variance = (&x * &x).mean_dim(-1, true, Kind::Float);
        let x = &x * (&variance + self.eps).rsqrt();
        (x * &self.weight).to_kind(dtype)
    }
}

// ============================================================================
// Linear layer
// ============================================================================

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, prefix, "weight")?,
            bias: get_weight_opt(weights, prefix, "bias"),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = x.matmul(&self.weight.tr());
        match &self.bias {
            Some(b) => out + b,
            None => out,
        }
    }
}

// ============================================================================
// Conv2d layer
// ============================================================================

pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: [i64; 2],
    pub padding: [i64; 2],
}

impl Conv2d {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        stride: [i64; 2],
        padding: [i64; 2],
    ) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, prefix, "weight")?,
            bias: get_weight_opt(weights, prefix, "bias"),
            stride,
            padding,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.conv2d(
            &self.weight,
            self.bias.as_ref(),
            &self.stride,
            &self.padding,
            &[1, 1], // dilation
            1,        // groups
        )
    }
}

// ============================================================================
// Audio encoder self-attention (bidirectional, with bias)
// ============================================================================

pub struct AudioAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl AudioAttention {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_heads: usize,
        d_model: usize,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        Ok(Self {
            q_proj: Linear::load(weights, &format!("{}.q_proj", prefix))?,
            k_proj: Linear::load(weights, &format!("{}.k_proj", prefix))?,
            v_proj: Linear::load(weights, &format!("{}.v_proj", prefix))?,
            out_proj: Linear::load(weights, &format!("{}.out_proj", prefix))?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let (bsz, seq_len, _) = x.size3().unwrap();
        let nh = self.num_heads as i64;
        let hd = self.head_dim as i64;

        let q = self.q_proj.forward(x).reshape([bsz, seq_len, nh, hd]).permute([0, 2, 1, 3]);
        let k = self.k_proj.forward(x).reshape([bsz, seq_len, nh, hd]).permute([0, 2, 1, 3]);
        let v = self.v_proj.forward(x).reshape([bsz, seq_len, nh, hd]).permute([0, 2, 1, 3]);

        let scale = (hd as f64).sqrt();
        let mut attn = q.matmul(&k.transpose(-2, -1)) / scale;

        if let Some(m) = mask {
            attn = attn + m;
        }

        let attn = attn.softmax(-1, Kind::Float);
        let out = attn.matmul(&v); // (bsz, nh, seq_len, hd)
        let out = out.permute([0, 2, 1, 3]).reshape([bsz, seq_len, nh * hd]);
        self.out_proj.forward(&out)
    }
}

// ============================================================================
// Audio encoder FFN
// ============================================================================

pub struct AudioFfn {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl AudioFfn {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(weights, &format!("{}.fc1", prefix))?,
            fc2: Linear::load(weights, &format!("{}.fc2", prefix))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.fc1.forward(x).gelu("none");
        self.fc2.forward(&x)
    }
}

// ============================================================================
// Audio encoder layer
// ============================================================================

pub struct AudioEncoderLayer {
    pub self_attn_layer_norm: LayerNorm,
    pub self_attn: AudioAttention,
    pub final_layer_norm: LayerNorm,
    pub ffn: AudioFfn,
}

impl AudioEncoderLayer {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_heads: usize,
        d_model: usize,
    ) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: LayerNorm::load(
                weights, &format!("{}.self_attn_layer_norm", prefix), 1e-5,
            )?,
            self_attn: AudioAttention::load(
                weights, &format!("{}.self_attn", prefix), num_heads, d_model,
            )?,
            final_layer_norm: LayerNorm::load(
                weights, &format!("{}.final_layer_norm", prefix), 1e-5,
            )?,
            ffn: AudioFfn::load(weights, prefix)?,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Pre-norm + self-attention + residual
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x);
        let x = self.self_attn.forward(&x, mask);
        let x = &x + residual;

        // Pre-norm + FFN + residual
        let residual = x.shallow_clone();
        let h = self.final_layer_norm.forward(&x);
        let h = self.ffn.forward(&h);
        h + residual
    }
}

// ============================================================================
// Text decoder attention (GQA with QK-norm and MRoPE)
// ============================================================================

pub struct TextAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl TextAttention {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::load(weights, &format!("{}.q_proj", prefix))?,
            k_proj: Linear::load(weights, &format!("{}.k_proj", prefix))?,
            v_proj: Linear::load(weights, &format!("{}.v_proj", prefix))?,
            o_proj: Linear::load(weights, &format!("{}.o_proj", prefix))?,
            q_norm: RmsNorm::load(weights, &format!("{}.q_norm", prefix), rms_norm_eps)?,
            k_norm: RmsNorm::load(weights, &format!("{}.k_norm", prefix), rms_norm_eps)?,
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }

    /// Forward pass with KV cache support.
    ///
    /// cos, sin: (seq_len, head_dim) for RoPE
    /// kv_cache: optional (past_key, past_value)
    /// mask: causal attention mask
    ///
    /// Returns: (output, (new_key, new_value))
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: Option<&(Tensor, Tensor)>,
        mask: Option<&Tensor>,
    ) -> (Tensor, (Tensor, Tensor)) {
        let (bsz, seq_len, _) = x.size3().unwrap();
        let nqh = self.num_q_heads as i64;
        let nkvh = self.num_kv_heads as i64;
        let hd = self.head_dim as i64;

        // Project Q, K, V
        let q = self.q_proj.forward(x).reshape([bsz, seq_len, nqh, hd]).transpose(1, 2);
        let k = self.k_proj.forward(x).reshape([bsz, seq_len, nkvh, hd]).transpose(1, 2);
        let v = self.v_proj.forward(x).reshape([bsz, seq_len, nkvh, hd]).transpose(1, 2);

        // Apply QK normalization (per-head RMSNorm)
        let q = self.apply_head_norm(&q, &self.q_norm);
        let k = self.apply_head_norm(&k, &self.k_norm);

        // Apply RoPE
        let q = apply_rotary_emb(&q, cos, sin);
        let k = apply_rotary_emb(&k, cos, sin);

        // Update KV cache
        let (k, v) = if let Some((past_k, past_v)) = kv_cache {
            let k = Tensor::cat(&[past_k, &k], 2); // cat along seq dim
            let v = Tensor::cat(&[past_v, &v], 2);
            (k, v)
        } else {
            (k, v)
        };

        let new_cache = (k.shallow_clone(), v.shallow_clone());

        // Repeat KV heads to match Q heads
        let n_rep = self.num_q_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep);
        let v = repeat_kv(&v, n_rep);

        // Compute attention
        let scale = (hd as f64).sqrt();
        let mut attn_weights = q.matmul(&k.transpose(-2, -1)) / scale;

        if let Some(m) = mask {
            attn_weights = attn_weights + m;
        }

        let attn_weights = attn_weights.softmax(-1, Kind::Float).to_kind(x.kind());
        let out = attn_weights.matmul(&v);

        // Reshape and project output
        let out = out.transpose(1, 2).reshape([bsz, seq_len, nqh * hd]);
        let out = self.o_proj.forward(&out);

        (out, new_cache)
    }

    fn apply_head_norm(&self, x: &Tensor, norm: &RmsNorm) -> Tensor {
        // x shape: (bsz, num_heads, seq_len, head_dim)
        // RMSNorm is applied per-head (on the last dim)
        norm.forward(x)
    }
}

/// Repeat KV heads to match the number of query heads.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Tensor {
    if n_rep == 1 {
        return x.shallow_clone();
    }
    let (bsz, num_kv_heads, seq_len, head_dim) = x.size4().unwrap();
    x.unsqueeze(2)
        .expand([bsz, num_kv_heads, n_rep as i64, seq_len, head_dim], false)
        .reshape([bsz, num_kv_heads * n_rep as i64, seq_len, head_dim])
}

/// Apply rotary position embeddings.
///
/// x: (bsz, num_heads, seq_len, head_dim)
/// cos, sin: (seq_len, head_dim)
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    // cos/sin: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    let cos = cos.unsqueeze(0).unsqueeze(0);
    let sin = sin.unsqueeze(0).unsqueeze(0);

    let x_rotated = rotate_half(x);
    x * &cos + x_rotated * &sin
}

/// Rotate half: split x into two halves and swap with negation.
/// [x1, x2] -> [-x2, x1]
fn rotate_half(x: &Tensor) -> Tensor {
    let half = x.size().last().unwrap() / 2;
    let x1 = x.narrow(-1, 0, half);
    let x2 = x.narrow(-1, half, half);
    Tensor::cat(&[&(-&x2), &x1], -1)
}

// ============================================================================
// Text decoder MLP (SwiGLU)
// ============================================================================

pub struct TextMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl TextMlp {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::load(weights, &format!("{}.gate_proj", prefix))?,
            up_proj: Linear::load(weights, &format!("{}.up_proj", prefix))?,
            down_proj: Linear::load(weights, &format!("{}.down_proj", prefix))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(x).silu();
        let up = self.up_proj.forward(x);
        self.down_proj.forward(&(gate * up))
    }
}

// ============================================================================
// Text decoder layer
// ============================================================================

pub struct TextDecoderLayer {
    pub input_layernorm: RmsNorm,
    pub self_attn: TextAttention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: TextMlp,
}

impl TextDecoderLayer {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::load(
                weights, &format!("{}.input_layernorm", prefix), rms_norm_eps,
            )?,
            self_attn: TextAttention::load(
                weights,
                &format!("{}.self_attn", prefix),
                num_q_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
            )?,
            post_attention_layernorm: RmsNorm::load(
                weights, &format!("{}.post_attention_layernorm", prefix), rms_norm_eps,
            )?,
            mlp: TextMlp::load(weights, &format!("{}.mlp", prefix))?,
        })
    }

    /// Forward pass with KV cache.
    /// Returns (output, new_kv_cache)
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: Option<&(Tensor, Tensor)>,
        mask: Option<&Tensor>,
    ) -> (Tensor, (Tensor, Tensor)) {
        // Pre-norm + self-attention + residual
        let residual = x;
        let h = self.input_layernorm.forward(x);
        let (h, new_cache) = self.self_attn.forward(&h, cos, sin, kv_cache, mask);
        let x = &h + residual;

        // Pre-norm + MLP + residual
        let residual = x.shallow_clone();
        let h = self.post_attention_layernorm.forward(&x);
        let h = self.mlp.forward(&h);
        let out = h + residual;

        (out, new_cache)
    }
}

// ============================================================================
// MRoPE (Multimodal Rotary Position Embedding) utilities
// ============================================================================

/// Compute MRoPE cos/sin tensors for the given 3D position IDs.
///
/// position_ids: 3 vectors (temporal, height, width), each of length seq_len
/// Returns: (cos, sin) each of shape (seq_len, head_dim)
pub fn compute_mrope_cos_sin(
    position_ids: &[Vec<i64>; 3],
    head_dim: usize,
    rope_theta: f64,
    mrope_section: &[usize],
    interleaved: bool,
    device: tch::Device,
) -> (Tensor, Tensor) {
    let half_dim = head_dim / 2;
    let seq_len = position_ids[0].len();

    // Compute inverse frequencies
    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    // Build dim_map: for each freq index (0..half_dim), which MRoPE dim to use (0,1,2)
    let dim_map = if interleaved {
        build_interleaved_dim_map(mrope_section, half_dim)
    } else {
        build_contiguous_dim_map(mrope_section, half_dim)
    };

    // Compute cos/sin for each position
    let mut cos_vals = vec![0.0f32; seq_len * head_dim];
    let mut sin_vals = vec![0.0f32; seq_len * head_dim];

    for t in 0..seq_len {
        for j in 0..half_dim {
            let dim = dim_map[j];
            let pos = position_ids[dim][t] as f64;
            let angle = pos * inv_freq[j];
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            // First half
            cos_vals[t * head_dim + j] = c;
            sin_vals[t * head_dim + j] = s;
            // Second half (identical, standard RoPE doubling)
            cos_vals[t * head_dim + j + half_dim] = c;
            sin_vals[t * head_dim + j + half_dim] = s;
        }
    }

    let cos = Tensor::from_slice(&cos_vals)
        .reshape([seq_len as i64, head_dim as i64])
        .to_device(device);
    let sin = Tensor::from_slice(&sin_vals)
        .reshape([seq_len as i64, head_dim as i64])
        .to_device(device);

    (cos, sin)
}

/// Build a contiguous dim map: [0,0,...,0, 1,1,...,1, 2,2,...,2]
fn build_contiguous_dim_map(sections: &[usize], total: usize) -> Vec<usize> {
    let mut map = Vec::with_capacity(total);
    for (dim, &size) in sections.iter().enumerate() {
        for _ in 0..size {
            if map.len() >= total {
                break;
            }
            map.push(dim);
        }
    }
    // Pad with last dim if needed
    while map.len() < total {
        map.push(sections.len() - 1);
    }
    map
}

/// Build an interleaved dim map: [0,1,2, 0,1,2, ..., 0,0,0,0]
/// Cycles through dimensions round-robin until each dim's quota is filled.
fn build_interleaved_dim_map(sections: &[usize], total: usize) -> Vec<usize> {
    let n_dims = sections.len();
    let mut map = Vec::with_capacity(total);
    let mut counts = vec![0usize; n_dims];

    while map.len() < total {
        let prev_len = map.len();
        for dim in 0..n_dims {
            if map.len() >= total {
                break;
            }
            if counts[dim] < sections[dim] {
                map.push(dim);
                counts[dim] += 1;
            }
        }
        if map.len() == prev_len {
            break; // no progress (shouldn't happen if sum(sections) >= total)
        }
    }

    map
}
