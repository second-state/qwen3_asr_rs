use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Tensor};

/// Load all tensors from a safetensors file, converting bf16 to f32.
pub fn load_safetensors(path: &Path, device: Device) -> Result<HashMap<String, Tensor>> {
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read safetensors: {:?}", path))?;
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .with_context(|| format!("Failed to deserialize safetensors: {:?}", path))?;

    let mut result = HashMap::new();

    for (name, view) in tensors.iter() {
        let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
        let tensor = match view.dtype() {
            safetensors::Dtype::BF16 => {
                let f32_data = bf16_bytes_to_f32(view.data());
                Tensor::from_slice(&f32_data)
                    .reshape(&shape)
                    .to_device(device)
            }
            safetensors::Dtype::F16 => {
                let f32_data = f16_bytes_to_f32(view.data());
                Tensor::from_slice(&f32_data)
                    .reshape(&shape)
                    .to_device(device)
            }
            safetensors::Dtype::F32 => {
                let f32_data: Vec<f32> = view
                    .data()
                    .chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Tensor::from_slice(&f32_data)
                    .reshape(&shape)
                    .to_device(device)
            }
            safetensors::Dtype::I64 => {
                let i64_data: Vec<i64> = view
                    .data()
                    .chunks(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Tensor::from_slice(&i64_data)
                    .reshape(&shape)
                    .to_device(device)
            }
            dt => anyhow::bail!("Unsupported dtype in safetensors: {:?}", dt),
        };
        result.insert(name.to_string(), tensor);
    }

    Ok(result)
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            // BF16 to F32: shift mantissa+exponent left by 16 bits
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            half_to_float(bits)
        })
        .collect()
}

fn half_to_float(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = exponent;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e = e.wrapping_sub(1);
            }
            m &= 0x3FF;
            let e = (127u32 - 15 + 1).wrapping_add(e);
            f32::from_bits((sign << 31) | (e << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13))
    } else {
        let e = exponent + (127 - 15);
        f32::from_bits((sign << 31) | (e << 23) | (mantissa << 13))
    }
}

/// Get a tensor from the weights map with a given prefix and suffix.
pub fn get_weight(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    name: &str,
) -> Result<Tensor> {
    let key = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", prefix, name)
    };
    weights
        .get(&key)
        .map(|t| t.shallow_clone())
        .with_context(|| format!("Weight not found: {}", key))
}

/// Get an optional tensor (returns None if not found).
pub fn get_weight_opt(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    name: &str,
) -> Option<Tensor> {
    let key = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", prefix, name)
    };
    weights.get(&key).map(|t| t.shallow_clone())
}
