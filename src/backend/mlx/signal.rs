//! Signal processing operations implemented via MLX primitives.

use super::array::MlxArray;
use super::ops;
use std::f64::consts::PI;

/// Create a Hann window of the given size.
pub fn hann_window(size: i32) -> MlxArray {
    let n: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let n = MlxArray::from_f32(&n, &[size]);
    let scale = MlxArray::scalar_f32(2.0 * PI as f32 / (size - 1) as f32);
    let phase = ops::multiply(&n, &scale);
    let cos_phase = ops::cos(&phase);
    let one = MlxArray::scalar_f32(1.0);
    let half = MlxArray::scalar_f32(0.5);
    let diff = ops::subtract(&one, &cos_phase);
    ops::multiply(&half, &diff)
}

/// Reflection padding for 1D signals.
pub fn reflection_pad1d(x: &MlxArray, pad_left: i32, pad_right: i32) -> MlxArray {
    let shape = x.shape();
    let ndim = shape.len() as i32;
    let t = *shape.last().unwrap();
    let last_axis = ndim - 1;

    let mut parts = Vec::new();

    if pad_left > 0 {
        let indices: Vec<i32> = (1..=pad_left).rev().collect();
        let idx = MlxArray::from_i32(&indices, &[pad_left]);
        let left = ops::take(x, &idx, last_axis);
        parts.push(left);
    }

    parts.push(x.clone());

    if pad_right > 0 {
        let indices: Vec<i32> = (0..pad_right).map(|i| t - 2 - i).collect();
        let idx = MlxArray::from_i32(&indices, &[pad_right]);
        let right = ops::take(x, &idx, last_axis);
        parts.push(right);
    }

    let refs: Vec<&MlxArray> = parts.iter().collect();
    ops::concatenate(&refs, last_axis)
}

/// Constant padding for N-dimensional tensors.
pub fn constant_pad_nd(x: &MlxArray, pad_widths: &[i32], val: f32) -> MlxArray {
    let ndim = x.ndim();
    let val_arr = MlxArray::scalar_f32(val);

    let num_dim_pairs = pad_widths.len() / 2;
    let mut axes = Vec::with_capacity(num_dim_pairs);
    let mut low = Vec::with_capacity(num_dim_pairs);
    let mut high = Vec::with_capacity(num_dim_pairs);

    for i in 0..num_dim_pairs {
        let dim = ndim - 1 - i as i32;
        let pad_before = pad_widths[i * 2];
        let pad_after = pad_widths[i * 2 + 1];
        if pad_before != 0 || pad_after != 0 {
            axes.push(dim);
            low.push(pad_before);
            high.push(pad_after);
        }
    }

    if axes.is_empty() {
        return x.clone();
    }

    ops::pad(x, &axes, &low, &high, &val_arr)
}

/// Short-Time Fourier Transform magnitude.
///
/// Returns shape (n_frames, n_fft/2+1) as float32 abs values.
pub fn stft_magnitude(
    signal: &MlxArray,
    n_fft: i32,
    hop_length: i32,
    window: &MlxArray,
) -> MlxArray {
    let padded_len = signal.shape()[0];
    let n_frames = (padded_len - n_fft) / hop_length + 1;

    let mut frame_arrays = Vec::with_capacity(n_frames as usize);
    for i in 0..n_frames {
        let start = i * hop_length;
        let starts = vec![start];
        let stops = vec![start + n_fft];
        let strides = vec![1i32];
        let frame = ops::slice(signal, &starts, &stops, &strides);

        let windowed = ops::multiply(&frame, window);
        let spectrum = ops::rfft(&windowed, n_fft, -1);
        let magnitude = ops::abs(&spectrum);

        frame_arrays.push(magnitude);
    }

    let refs: Vec<&MlxArray> = frame_arrays.iter().collect();
    ops::stack(&refs, 0)
}
