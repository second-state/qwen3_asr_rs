//! Safe wrappers for MLX array operations.
#![allow(dead_code)]

use super::array::MlxArray;
use super::ffi;
use super::stream::default_stream;

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

pub fn add(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_add(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn subtract(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_subtract(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn multiply(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_multiply(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn divide(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_divide(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn negative(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_negative(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn abs(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_abs(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn power(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_power(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn maximum(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_maximum(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn minimum(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_minimum(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn clip(a: &MlxArray, min: &MlxArray, max: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_clip(&mut res.ptr, a.ptr, min.ptr, max.ptr, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Matrix multiplication
// ---------------------------------------------------------------------------

pub fn matmul(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_matmul(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

pub fn reshape(a: &MlxArray, shape: &[i32]) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_reshape(&mut res.ptr, a.ptr, shape.as_ptr(), shape.len(), default_stream());
    }
    res
}

pub fn transpose(a: &MlxArray, axes: &[i32]) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_transpose_axes(
            &mut res.ptr, a.ptr, axes.as_ptr(), axes.len(), default_stream(),
        );
    }
    res
}

pub fn swapaxes(a: &MlxArray, axis1: i32, axis2: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_swapaxes(&mut res.ptr, a.ptr, axis1, axis2, default_stream()) };
    res
}

pub fn expand_dims(a: &MlxArray, axes: &[i32]) -> MlxArray {
    let mut result = a.clone();
    for &axis in axes {
        let mut res = MlxArray::empty();
        unsafe { ffi::mlx_expand_dims(&mut res.ptr, result.ptr, axis, default_stream()) };
        result = res;
    }
    result
}

pub fn squeeze(a: &MlxArray, axes: &[i32]) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_squeeze_axes(
            &mut res.ptr, a.ptr, axes.as_ptr(), axes.len(), default_stream(),
        );
    }
    res
}

pub fn slice(a: &MlxArray, start: &[i32], stop: &[i32], strides: &[i32]) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_slice(
            &mut res.ptr,
            a.ptr,
            start.as_ptr(), start.len(),
            stop.as_ptr(), stop.len(),
            strides.as_ptr(), strides.len(),
            default_stream(),
        );
    }
    res
}

pub fn broadcast_to(a: &MlxArray, shape: &[i32]) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_broadcast_to(
            &mut res.ptr, a.ptr, shape.as_ptr(), shape.len(), default_stream(),
        );
    }
    res
}

pub fn flatten(a: &MlxArray, start_axis: i32, end_axis: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_flatten(&mut res.ptr, a.ptr, start_axis, end_axis, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Concatenation
// ---------------------------------------------------------------------------

pub fn concatenate(arrays: &[&MlxArray], axis: i32) -> MlxArray {
    let vec = unsafe { ffi::mlx_vector_array_new() };
    for a in arrays {
        unsafe { ffi::mlx_vector_array_append_value(vec, a.ptr) };
    }
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_concatenate_axis(&mut res.ptr, vec, axis, default_stream()) };
    unsafe { ffi::mlx_vector_array_free(vec) };
    res
}

pub fn stack(arrays: &[&MlxArray], axis: i32) -> MlxArray {
    let vec = unsafe { ffi::mlx_vector_array_new() };
    for a in arrays {
        unsafe { ffi::mlx_vector_array_append_value(vec, a.ptr) };
    }
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_stack_axis(&mut res.ptr, vec, axis, default_stream()) };
    unsafe { ffi::mlx_vector_array_free(vec) };
    res
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

pub fn take(a: &MlxArray, indices: &MlxArray, axis: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_take_axis(&mut res.ptr, a.ptr, indices.ptr, axis, default_stream()) };
    res
}

pub fn take_along_axis(a: &MlxArray, indices: &MlxArray, axis: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_take_along_axis(&mut res.ptr, a.ptr, indices.ptr, axis, default_stream());
    }
    res
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

pub fn sum(a: &MlxArray, axes: &[i32], keepdims: bool) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_sum_axes(
            &mut res.ptr, a.ptr, axes.as_ptr(), axes.len(), keepdims, default_stream(),
        );
    }
    res
}

pub fn mean(a: &MlxArray, axes: &[i32], keepdims: bool) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_mean_axes(
            &mut res.ptr, a.ptr, axes.as_ptr(), axes.len(), keepdims, default_stream(),
        );
    }
    res
}

pub fn var(a: &MlxArray, axes: &[i32], keepdims: bool, ddof: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_var_axes(
            &mut res.ptr, a.ptr, axes.as_ptr(), axes.len(), keepdims, ddof, default_stream(),
        );
    }
    res
}

pub fn mean_all(a: &MlxArray, keepdims: bool) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_mean(&mut res.ptr, a.ptr, keepdims, default_stream()) };
    res
}

pub fn max_all(a: &MlxArray, keepdims: bool) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_max(&mut res.ptr, a.ptr, keepdims, default_stream()) };
    res
}

pub fn argmax(a: &MlxArray, axis: i32, keepdims: bool) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_argmax_axis(&mut res.ptr, a.ptr, axis, keepdims, default_stream()) };
    res
}

pub fn argmin(a: &MlxArray, axis: i32, keepdims: bool) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_argmin_axis(&mut res.ptr, a.ptr, axis, keepdims, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Math functions
// ---------------------------------------------------------------------------

pub fn exp(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_exp(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn log(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_log(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn sqrt(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_sqrt(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn rsqrt(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_rsqrt(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn sin(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_sin(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn cos(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_cos(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn sigmoid(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_sigmoid(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn tanh(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_tanh(&mut res.ptr, a.ptr, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

pub fn softmax(a: &MlxArray, axes: &[i32]) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_softmax_axes(
            &mut res.ptr, a.ptr, axes.as_ptr(), axes.len(), false, default_stream(),
        );
    }
    res
}

pub fn silu(a: &MlxArray) -> MlxArray {
    let sig = sigmoid(a);
    multiply(a, &sig)
}

pub fn gelu(a: &MlxArray) -> MlxArray {
    let coeff = MlxArray::scalar_f32(1.702);
    let scaled = multiply(a, &coeff);
    let sig = sigmoid(&scaled);
    multiply(a, &sig)
}

// ---------------------------------------------------------------------------
// Comparison / logical
// ---------------------------------------------------------------------------

pub fn less(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_less(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn greater(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_greater(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn equal(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_equal(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn logical_or(a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_logical_or(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn logical_not(a: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_logical_not(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn where_cond(cond: &MlxArray, x: &MlxArray, y: &MlxArray) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_where(&mut res.ptr, cond.ptr, x.ptr, y.ptr, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Triangular
// ---------------------------------------------------------------------------

pub fn triu(a: &MlxArray, k: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_triu(&mut res.ptr, a.ptr, k, default_stream()) };
    res
}

pub fn tril(a: &MlxArray, k: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_tril(&mut res.ptr, a.ptr, k, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------

pub fn conv1d(
    input: &MlxArray,
    weight: &MlxArray,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_conv1d(
            &mut res.ptr, input.ptr, weight.ptr,
            stride, padding, dilation, groups, default_stream(),
        );
    }
    res
}

pub fn conv2d(
    input: &MlxArray,
    weight: &MlxArray,
    stride: [i32; 2],
    padding: [i32; 2],
    dilation: [i32; 2],
    groups: i32,
) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_conv2d(
            &mut res.ptr, input.ptr, weight.ptr,
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            groups, default_stream(),
        );
    }
    res
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

pub fn pad(
    a: &MlxArray,
    axes: &[i32],
    low_pad: &[i32],
    high_pad: &[i32],
    val: &MlxArray,
) -> MlxArray {
    let mut res = MlxArray::empty();
    let mode = b"constant\0".as_ptr() as *const std::os::raw::c_char;
    unsafe {
        ffi::mlx_pad(
            &mut res.ptr, a.ptr,
            axes.as_ptr(), axes.len(),
            low_pad.as_ptr(), low_pad.len(),
            high_pad.as_ptr(), high_pad.len(),
            val.ptr, mode, default_stream(),
        );
    }
    res
}

// ---------------------------------------------------------------------------
// Fast ML ops
// ---------------------------------------------------------------------------

pub fn fast_rms_norm(x: &MlxArray, weight: &MlxArray, eps: f32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe {
        ffi::mlx_fast_rms_norm(&mut res.ptr, x.ptr, weight.ptr, eps, default_stream());
    }
    res
}

pub fn fast_layer_norm(
    x: &MlxArray,
    weight: &MlxArray,
    bias: Option<&MlxArray>,
    eps: f32,
) -> MlxArray {
    let mut res = MlxArray::empty();
    let bias_ptr = bias.map_or(std::ptr::null_mut(), |b| b.ptr);
    unsafe {
        ffi::mlx_fast_layer_norm(&mut res.ptr, x.ptr, weight.ptr, bias_ptr, eps, default_stream());
    }
    res
}

// ---------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------

pub fn rfft(a: &MlxArray, n: i32, axis: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_fft_rfft(&mut res.ptr, a.ptr, n, axis, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Top-k and sorting
// ---------------------------------------------------------------------------

pub fn topk(a: &MlxArray, k: i32, axis: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_topk_axis(&mut res.ptr, a.ptr, k, axis, default_stream()) };
    res
}

pub fn argsort(a: &MlxArray, axis: i32) -> MlxArray {
    let mut res = MlxArray::empty();
    unsafe { ffi::mlx_argsort_axis(&mut res.ptr, a.ptr, axis, default_stream()) };
    res
}
