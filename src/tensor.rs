//! Unified tensor abstraction over tch (libtorch) and MLX backends.
//!
//! All neural network modules use these types instead of importing `tch` directly.

// ---------------------------------------------------------------------------
// DType — data type abstraction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Int64,
    Int32,
    Bool,
}

#[cfg(feature = "tch-backend")]
impl From<DType> for tch::Kind {
    fn from(dt: DType) -> Self {
        match dt {
            DType::Float32 => tch::Kind::Float,
            DType::Float16 => tch::Kind::Half,
            DType::BFloat16 => tch::Kind::BFloat16,
            DType::Int64 => tch::Kind::Int64,
            DType::Int32 => tch::Kind::Int,
            DType::Bool => tch::Kind::Bool,
        }
    }
}

#[cfg(feature = "tch-backend")]
impl From<tch::Kind> for DType {
    fn from(kind: tch::Kind) -> Self {
        match kind {
            tch::Kind::Float => DType::Float32,
            tch::Kind::Half => DType::Float16,
            tch::Kind::BFloat16 => DType::BFloat16,
            tch::Kind::Int64 => DType::Int64,
            tch::Kind::Int => DType::Int32,
            tch::Kind::Bool => DType::Bool,
            _ => DType::Float32,
        }
    }
}

#[cfg(feature = "mlx")]
impl From<DType> for crate::backend::mlx::ffi::mlx_dtype {
    fn from(dt: DType) -> Self {
        use crate::backend::mlx::ffi::mlx_dtype::*;
        match dt {
            DType::Float32 => MLX_FLOAT32,
            DType::Float16 => MLX_FLOAT16,
            DType::BFloat16 => MLX_BFLOAT16,
            DType::Int64 => MLX_INT64,
            DType::Int32 => MLX_INT32,
            DType::Bool => MLX_BOOL,
        }
    }
}

#[cfg(feature = "mlx")]
impl From<crate::backend::mlx::ffi::mlx_dtype> for DType {
    fn from(dt: crate::backend::mlx::ffi::mlx_dtype) -> Self {
        use crate::backend::mlx::ffi::mlx_dtype::*;
        match dt {
            MLX_FLOAT32 | MLX_FLOAT64 => DType::Float32,
            MLX_FLOAT16 => DType::Float16,
            MLX_BFLOAT16 => DType::BFloat16,
            MLX_INT64 => DType::Int64,
            MLX_INT32 | MLX_INT8 | MLX_INT16 => DType::Int32,
            MLX_BOOL => DType::Bool,
            _ => DType::Float32,
        }
    }
}

// ---------------------------------------------------------------------------
// Device — compute device abstraction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu(usize),
}

impl Device {
    pub fn gpu() -> Self {
        Device::Gpu(0)
    }
}

#[cfg(feature = "tch-backend")]
impl From<Device> for tch::Device {
    fn from(d: Device) -> Self {
        match d {
            Device::Cpu => tch::Device::Cpu,
            Device::Gpu(i) => tch::Device::Cuda(i),
        }
    }
}

#[cfg(feature = "tch-backend")]
impl From<tch::Device> for Device {
    fn from(d: tch::Device) -> Self {
        match d {
            tch::Device::Cpu => Device::Cpu,
            tch::Device::Cuda(i) => Device::Gpu(i),
            _ => Device::Cpu,
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor — unified tensor type
// ---------------------------------------------------------------------------

pub struct Tensor {
    #[cfg(feature = "tch-backend")]
    pub(crate) inner: tch::Tensor,

    #[cfg(feature = "mlx")]
    pub(crate) inner: crate::backend::mlx::array::MlxArray,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?}, dtype={:?})", self.size(), self.kind())
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        #[cfg(feature = "tch-backend")]
        { Tensor { inner: self.inner.shallow_clone() } }
        #[cfg(feature = "mlx")]
        { Tensor { inner: self.inner.clone() } }
    }
}

// ===== tch backend implementation =====

#[cfg(feature = "tch-backend")]
#[allow(dead_code)]
impl Tensor {
    pub fn from_tch(t: tch::Tensor) -> Self {
        Tensor { inner: t }
    }

    pub fn as_tch(&self) -> &tch::Tensor {
        &self.inner
    }

    pub fn into_tch(self) -> tch::Tensor {
        self.inner
    }

    // -- Creation --

    pub fn from_slice_f32(data: &[f32]) -> Self {
        Tensor::from_tch(tch::Tensor::from_slice(data))
    }

    pub fn from_slice_i64(data: &[i64]) -> Self {
        Tensor::from_tch(tch::Tensor::from_slice(data))
    }

    pub fn zeros(shape: &[i64], dtype: DType, device: Device) -> Self {
        let opts = (tch::Kind::from(dtype), tch::Device::from(device));
        Tensor::from_tch(tch::Tensor::zeros(shape, opts))
    }

    pub fn ones(shape: &[i64], dtype: DType, device: Device) -> Self {
        let opts = (tch::Kind::from(dtype), tch::Device::from(device));
        Tensor::from_tch(tch::Tensor::ones(shape, opts))
    }

    pub fn full(shape: &[i64], val: f64, dtype: DType, device: Device) -> Self {
        let opts = (tch::Kind::from(dtype), tch::Device::from(device));
        Tensor::from_tch(tch::Tensor::full(shape, val, opts))
    }

    pub fn arange(start: i64, end: i64, device: Device) -> Self {
        let t = tch::Tensor::arange(end - start, (tch::Kind::Int64, tch::Device::from(device)))
            + start;
        Tensor::from_tch(t)
    }

    pub fn arange_f(start: f64, end: f64, step: f64, dtype: DType, device: Device) -> Self {
        let t = tch::Tensor::arange_start_step(
            start, end, step,
            (tch::Kind::from(dtype), tch::Device::from(device)),
        );
        Tensor::from_tch(t)
    }

    pub fn cat(tensors: &[Tensor], dim: i64) -> Self {
        let inner: Vec<&tch::Tensor> = tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_tch(tch::Tensor::cat(&inner, dim))
    }

    pub fn stack(tensors: &[Tensor], dim: i64) -> Self {
        let inner: Vec<&tch::Tensor> = tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_tch(tch::Tensor::stack(&inner, dim))
    }

    pub fn embedding(weight: &Tensor, indices: &Tensor) -> Self {
        Tensor::from_tch(tch::Tensor::embedding(
            &weight.inner, &indices.inner, -1, false, false,
        ))
    }

    pub fn hann_window(size: i64, device: Device) -> Self {
        Tensor::from_tch(tch::Tensor::hann_window(
            size, (tch::Kind::Float, tch::Device::from(device)),
        ))
    }

    // -- Shape --

    pub fn size(&self) -> Vec<i64> {
        self.inner.size()
    }

    pub fn size3(&self) -> (i64, i64, i64) {
        self.inner.size3().unwrap()
    }

    pub fn size4(&self) -> (i64, i64, i64, i64) {
        self.inner.size4().unwrap()
    }

    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    pub fn view(&self, shape: &[i64]) -> Self {
        Tensor::from_tch(self.inner.view(shape))
    }

    pub fn reshape(&self, shape: &[i64]) -> Self {
        Tensor::from_tch(self.inner.reshape(shape))
    }

    pub fn narrow(&self, dim: i64, start: i64, len: i64) -> Self {
        Tensor::from_tch(self.inner.narrow(dim, start, len))
    }

    pub fn unsqueeze(&self, dim: i64) -> Self {
        Tensor::from_tch(self.inner.unsqueeze(dim))
    }

    pub fn squeeze_dim(&self, dim: i64) -> Self {
        Tensor::from_tch(self.inner.squeeze_dim(dim))
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        Tensor::from_tch(self.inner.transpose(dim0, dim1))
    }

    pub fn permute(&self, dims: &[i64]) -> Self {
        Tensor::from_tch(self.inner.permute(dims))
    }

    pub fn expand(&self, size: &[i64], implicit: bool) -> Self {
        Tensor::from_tch(self.inner.expand(size, implicit))
    }

    pub fn contiguous(&self) -> Self {
        Tensor::from_tch(self.inner.contiguous())
    }

    pub fn tr(&self) -> Self {
        Tensor::from_tch(self.inner.tr())
    }

    pub fn get(&self, index: i64) -> Self {
        Tensor::from_tch(self.inner.get(index))
    }

    pub fn select(&self, dim: i64, index: i64) -> Self {
        Tensor::from_tch(self.inner.select(dim, index))
    }

    // -- Arithmetic --

    pub fn matmul(&self, other: &Tensor) -> Self {
        Tensor::from_tch(self.inner.matmul(&other.inner))
    }

    pub fn pow_scalar(&self, exp: f64) -> Self {
        Tensor::from_tch(self.inner.pow_tensor_scalar(exp))
    }

    pub fn neg(&self) -> Self {
        Tensor::from_tch(self.inner.neg())
    }

    pub fn clamp_min(&self, min: f64) -> Self {
        Tensor::from_tch(self.inner.clamp_min(min))
    }

    pub fn maximum(&self, other: &Tensor) -> Self {
        Tensor::from_tch(self.inner.maximum(&other.inner))
    }

    // -- Math --

    pub fn abs(&self) -> Self {
        Tensor::from_tch(self.inner.abs())
    }

    pub fn square(&self) -> Self {
        Tensor::from_tch(self.inner.square())
    }

    pub fn sqrt(&self) -> Self {
        Tensor::from_tch(self.inner.sqrt())
    }

    pub fn rsqrt(&self) -> Self {
        let s = self.inner.sqrt();
        Tensor::from_tch(s.reciprocal())
    }

    pub fn log10(&self) -> Self {
        Tensor::from_tch(self.inner.log10())
    }

    pub fn sin(&self) -> Self {
        Tensor::from_tch(self.inner.sin())
    }

    pub fn cos(&self) -> Self {
        Tensor::from_tch(self.inner.cos())
    }

    pub fn exp(&self) -> Self {
        Tensor::from_tch(self.inner.exp())
    }

    // -- Activations --

    pub fn softmax(&self, dim: i64) -> Self {
        Tensor::from_tch(self.inner.softmax(dim, tch::Kind::Float))
    }

    pub fn gelu(&self) -> Self {
        Tensor::from_tch(self.inner.gelu("none"))
    }

    pub fn silu(&self) -> Self {
        Tensor::from_tch(self.inner.silu())
    }

    // -- Reduction --

    pub fn mean_dim(&self, dims: &[i64], keepdim: bool) -> Self {
        Tensor::from_tch(self.inner.mean_dim(dims, keepdim, tch::Kind::Float))
    }

    pub fn max(&self) -> Self {
        Tensor::from_tch(self.inner.max())
    }

    // -- Indexing --

    pub fn argmax(&self, dim: i64, keepdim: bool) -> Self {
        Tensor::from_tch(self.inner.argmax(dim, keepdim))
    }

    pub fn triu(&self, diagonal: i64) -> Self {
        Tensor::from_tch(self.inner.triu(diagonal))
    }

    pub fn slice_scatter(&self, src: &Tensor, dim: i64, start: i64, end: i64, step: i64) -> Self {
        Tensor::from_tch(self.inner.slice_scatter(&src.inner, dim, Some(start), Some(end), step))
    }

    pub fn fill_(&mut self, val: f64) {
        let _ = self.inner.fill_(val);
    }

    // -- Normalization --

    pub fn layer_norm(
        &self,
        normalized_shape: &[i64],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Self {
        Tensor::from_tch(self.inner.layer_norm(
            normalized_shape,
            weight.map(|w| &w.inner),
            bias.map(|b| &b.inner),
            eps,
            true,
        ))
    }

    // -- Convolution --

    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self {
        let bias_inner = bias.map(|b| &b.inner);
        Tensor::from_tch(self.inner.conv2d(
            &weight.inner, bias_inner, stride, padding, dilation, groups,
        ))
    }

    // -- Signal --

    pub fn reflection_pad1d(&self, pad: &[i64]) -> Self {
        Tensor::from_tch(self.inner.reflection_pad1d(pad))
    }

    pub fn stft(
        &self,
        n_fft: i64,
        hop_length: i64,
        win_length: i64,
        window: &Tensor,
        normalized: bool,
        onesided: bool,
        return_complex: bool,
    ) -> Self {
        Tensor::from_tch(self.inner.stft(
            n_fft,
            Some(hop_length),
            Some(win_length),
            Some(&window.inner),
            normalized,
            onesided,
            return_complex,
            false,
        ))
    }

    // -- Type / Device --

    pub fn to_dtype(&self, dtype: DType) -> Self {
        Tensor::from_tch(self.inner.to_kind(tch::Kind::from(dtype)))
    }

    pub fn to_device(&self, device: Device) -> Self {
        Tensor::from_tch(self.inner.to_device(tch::Device::from(device)))
    }

    pub fn kind(&self) -> DType {
        DType::from(self.inner.kind())
    }

    pub fn device(&self) -> Device {
        Device::from(self.inner.device())
    }

    pub fn shallow_clone(&self) -> Self {
        Tensor::from_tch(self.inner.shallow_clone())
    }

    // -- Data extraction --

    pub fn int64_value(&self, indices: &[i64]) -> i64 {
        self.inner.int64_value(indices)
    }

    pub fn f64_value(&self, indices: &[i64]) -> f64 {
        self.inner.double_value(indices)
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        let flat = self.inner.view(-1);
        let numel = flat.numel();
        let mut result = vec![0.0f32; numel];
        flat.to_kind(tch::Kind::Float).copy_data(&mut result, numel);
        result
    }
}

// ===== MLX backend implementation =====

#[cfg(feature = "mlx")]
#[allow(dead_code)]
impl Tensor {
    pub fn from_mlx(a: crate::backend::mlx::array::MlxArray) -> Self {
        Tensor { inner: a }
    }

    pub fn as_mlx(&self) -> &crate::backend::mlx::array::MlxArray {
        &self.inner
    }

    // -- Creation --

    pub fn from_slice_f32(data: &[f32]) -> Self {
        let shape = [data.len() as i32];
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::from_f32(data, &shape))
    }

    pub fn from_slice_i64(data: &[i64]) -> Self {
        let shape = [data.len() as i32];
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::from_i64(data, &shape))
    }

    pub fn zeros(shape: &[i64], dtype: DType, _device: Device) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::zeros(&shape_i32, dtype.into()))
    }

    pub fn ones(shape: &[i64], dtype: DType, _device: Device) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::ones(&shape_i32, dtype.into()))
    }

    pub fn full(shape: &[i64], val: f64, dtype: DType, _device: Device) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let val_arr = crate::backend::mlx::array::MlxArray::scalar_f32(val as f32);
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::full(&shape_i32, &val_arr, dtype.into()))
    }

    pub fn arange(start: i64, end: i64, _device: Device) -> Self {
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::arange(
            start as f64, end as f64, 1.0,
            crate::backend::mlx::ffi::mlx_dtype::MLX_INT64,
        ))
    }

    pub fn arange_f(start: f64, end: f64, step: f64, dtype: DType, _device: Device) -> Self {
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::arange(start, end, step, dtype.into()))
    }

    pub fn cat(tensors: &[Tensor], dim: i64) -> Self {
        let refs: Vec<&crate::backend::mlx::array::MlxArray> =
            tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::concatenate(&refs, dim as i32))
    }

    pub fn stack(tensors: &[Tensor], dim: i64) -> Self {
        let refs: Vec<&crate::backend::mlx::array::MlxArray> =
            tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::stack(&refs, dim as i32))
    }

    pub fn embedding(weight: &Tensor, indices: &Tensor) -> Self {
        // Embedding is just take(weight, indices, axis=0)
        Tensor::from_mlx(crate::backend::mlx::ops::take(&weight.inner, &indices.inner, 0))
    }

    pub fn hann_window(size: i64, _device: Device) -> Self {
        Tensor::from_mlx(crate::backend::mlx::signal::hann_window(size as i32))
    }

    // -- Shape --

    pub fn size(&self) -> Vec<i64> {
        self.inner.shape().iter().map(|&s| s as i64).collect()
    }

    pub fn size3(&self) -> (i64, i64, i64) {
        let s = self.size();
        (s[0], s[1], s[2])
    }

    pub fn size4(&self) -> (i64, i64, i64, i64) {
        let s = self.size();
        (s[0], s[1], s[2], s[3])
    }

    pub fn dim(&self) -> usize {
        self.inner.ndim() as usize
    }

    pub fn view(&self, shape: &[i64]) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::reshape(&self.inner, &shape_i32))
    }

    pub fn reshape(&self, shape: &[i64]) -> Self {
        self.view(shape)
    }

    pub fn narrow(&self, dim: i64, start: i64, len: i64) -> Self {
        let ndim = self.inner.ndim();
        let dim = if dim < 0 { ndim as i64 + dim } else { dim } as i32;
        let shape = self.inner.shape();
        let mut starts = vec![0i32; ndim as usize];
        let mut stops: Vec<i32> = shape.clone();
        let strides = vec![1i32; ndim as usize];
        starts[dim as usize] = start as i32;
        stops[dim as usize] = (start + len) as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::slice(&self.inner, &starts, &stops, &strides))
    }

    pub fn unsqueeze(&self, dim: i64) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim + 1
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::expand_dims(&self.inner, &[dim]))
    }

    pub fn squeeze_dim(&self, dim: i64) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::squeeze(&self.inner, &[dim]))
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        let ndim = self.inner.ndim();
        let dim0 = if dim0 < 0 { ndim as i64 + dim0 } else { dim0 } as i32;
        let dim1 = if dim1 < 0 { ndim as i64 + dim1 } else { dim1 } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::swapaxes(&self.inner, dim0, dim1))
    }

    pub fn permute(&self, dims: &[i64]) -> Self {
        let dims_i32: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::transpose(&self.inner, &dims_i32))
    }

    pub fn expand(&self, size: &[i64], _implicit: bool) -> Self {
        let current = self.inner.shape();
        let shape_i32: Vec<i32> = size
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                if s == -1 { current[i] } else { s as i32 }
            })
            .collect();
        Tensor::from_mlx(crate::backend::mlx::ops::broadcast_to(&self.inner, &shape_i32))
    }

    pub fn contiguous(&self) -> Self {
        self.clone()
    }

    pub fn tr(&self) -> Self {
        self.transpose(-2, -1)
    }

    pub fn get(&self, index: i64) -> Self {
        self.select(0, index)
    }

    pub fn select(&self, dim: i64, index: i64) -> Self {
        let idx = crate::backend::mlx::array::MlxArray::from_i32(&[index as i32], &[1]);
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        let taken = crate::backend::mlx::ops::take(&self.inner, &idx, dim);
        Tensor::from_mlx(crate::backend::mlx::ops::squeeze(&taken, &[dim]))
    }

    // -- Arithmetic --

    pub fn matmul(&self, other: &Tensor) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::matmul(&self.inner, &other.inner))
    }

    pub fn pow_scalar(&self, exp: f64) -> Self {
        let exp_arr = crate::backend::mlx::array::MlxArray::scalar_f32(exp as f32);
        Tensor::from_mlx(crate::backend::mlx::ops::power(&self.inner, &exp_arr))
    }

    pub fn neg(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::negative(&self.inner))
    }

    pub fn clamp_min(&self, min: f64) -> Self {
        let min_arr = crate::backend::mlx::array::MlxArray::scalar_f32(min as f32);
        Tensor::from_mlx(crate::backend::mlx::ops::maximum(&self.inner, &min_arr))
    }

    pub fn maximum(&self, other: &Tensor) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::maximum(&self.inner, &other.inner))
    }

    // -- Math --

    pub fn abs(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::abs(&self.inner))
    }

    pub fn square(&self) -> Self {
        self.pow_scalar(2.0)
    }

    pub fn sqrt(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::sqrt(&self.inner))
    }

    pub fn rsqrt(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::rsqrt(&self.inner))
    }

    pub fn log10(&self) -> Self {
        // log10(x) = ln(x) / ln(10)
        let ln_x = crate::backend::mlx::ops::log(&self.inner);
        let ln10 = crate::backend::mlx::array::MlxArray::scalar_f32(std::f32::consts::LN_10);
        Tensor::from_mlx(crate::backend::mlx::ops::divide(&ln_x, &ln10))
    }

    pub fn sin(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::sin(&self.inner))
    }

    pub fn cos(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::cos(&self.inner))
    }

    pub fn exp(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::exp(&self.inner))
    }

    // -- Activations --

    pub fn softmax(&self, dim: i64) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::softmax(&self.inner, &[dim]))
    }

    pub fn gelu(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::gelu(&self.inner))
    }

    pub fn silu(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::silu(&self.inner))
    }

    // -- Reduction --

    pub fn mean_dim(&self, dims: &[i64], keepdim: bool) -> Self {
        let dims_i32: Vec<i32> = dims.iter().map(|&d| {
            if d < 0 { self.inner.ndim() as i32 + d as i32 } else { d as i32 }
        }).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::mean(&self.inner, &dims_i32, keepdim))
    }

    pub fn max(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::max_all(&self.inner, false))
    }

    // -- Indexing --

    pub fn argmax(&self, dim: i64, keepdim: bool) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::argmax(&self.inner, dim, keepdim))
    }

    pub fn triu(&self, diagonal: i64) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::triu(&self.inner, diagonal as i32))
    }

    /// Replaces self[..., start:end:step, ...] along dim with src.
    pub fn slice_scatter(&self, src: &Tensor, dim: i64, start: i64, end: i64, _step: i64) -> Self {
        let ndim = self.inner.ndim() as usize;
        let dim = if dim < 0 { ndim as i64 + dim } else { dim } as usize;
        let shape = self.inner.shape();
        let dim_size = shape[dim] as i64;

        // Build: [before, src, after]
        let mut parts: Vec<Tensor> = Vec::new();

        if start > 0 {
            parts.push(self.narrow(dim as i64, 0, start));
        }
        parts.push(src.clone());
        let after_start = end;
        if after_start < dim_size {
            parts.push(self.narrow(dim as i64, after_start, dim_size - after_start));
        }

        if parts.len() == 1 {
            return parts.into_iter().next().unwrap();
        }

        Tensor::cat(&parts, dim as i64)
    }

    /// In-place fill (MLX: returns new tensor with fill value).
    /// Used for building attention masks.
    pub fn fill_(&self, val: f64) -> Self {
        let shape: Vec<i64> = self.size();
        Tensor::full(&shape, val, DType::Float32, Device::Gpu(0))
    }

    // -- Normalization --

    pub fn layer_norm(
        &self,
        _normalized_shape: &[i64],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Self {
        if let Some(w) = weight {
            Tensor::from_mlx(crate::backend::mlx::ops::fast_layer_norm(
                &self.inner,
                &w.inner,
                bias.map(|b| &b.inner),
                eps as f32,
            ))
        } else {
            let mean = self.mean_dim(&[-1], true);
            let var_t = {
                let diff = self - &mean;
                (&diff * &diff).mean_dim(&[-1], true)
            };
            let normalized = &(self - &mean) / &(&var_t + eps).sqrt();
            if let Some(b) = bias {
                &normalized + b
            } else {
                normalized
            }
        }
    }

    // -- Convolution --

    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self {
        // PyTorch: input [N, C, H, W], weight [C_out, C_in, kH, kW]
        // MLX:     input [N, H, W, C], weight [C_out, kH, kW, C_in]
        let input_t = self.permute(&[0, 2, 3, 1]); // [N, C, H, W] -> [N, H, W, C]
        let weight_t = weight.permute(&[0, 2, 3, 1]); // [C_out, C_in, kH, kW] -> [C_out, kH, kW, C_in]

        let result = crate::backend::mlx::ops::conv2d(
            &input_t.inner,
            &weight_t.inner,
            [stride[0] as i32, stride[1] as i32],
            [padding[0] as i32, padding[1] as i32],
            [dilation[0] as i32, dilation[1] as i32],
            groups as i32,
        );
        // Output: [N, H_out, W_out, C_out] -> [N, C_out, H_out, W_out]
        let out = Tensor::from_mlx(result).permute(&[0, 3, 1, 2]);
        if let Some(b) = bias {
            // bias is [C_out], reshape to [1, C_out, 1, 1] for broadcasting
            &out + &b.reshape(&[-1, 1, 1]).unsqueeze(0)
        } else {
            out
        }
    }

    // -- Signal --

    pub fn reflection_pad1d(&self, pad: &[i64]) -> Self {
        Tensor::from_mlx(crate::backend::mlx::signal::reflection_pad1d(
            &self.inner, pad[0] as i32, pad[1] as i32,
        ))
    }

    pub fn stft(
        &self,
        n_fft: i64,
        hop_length: i64,
        _win_length: i64,
        window: &Tensor,
        _normalized: bool,
        _onesided: bool,
        _return_complex: bool,
    ) -> Self {
        // stft_magnitude returns [n_frames, freq_bins].
        // Transpose to [freq_bins, n_frames] to match tch STFT output layout.
        let mag = crate::backend::mlx::signal::stft_magnitude(
            &self.inner,
            n_fft as i32,
            hop_length as i32,
            &window.inner,
        );
        Tensor::from_mlx(crate::backend::mlx::ops::swapaxes(&mag, 0, 1))
    }

    // -- Type / Device --

    pub fn to_dtype(&self, dtype: DType) -> Self {
        Tensor::from_mlx(self.inner.astype(dtype.into()))
    }

    pub fn to_device(&self, _device: Device) -> Self {
        self.clone()
    }

    pub fn kind(&self) -> DType {
        DType::from(self.inner.dtype())
    }

    pub fn device(&self) -> Device {
        Device::Gpu(0)
    }

    pub fn shallow_clone(&self) -> Self {
        self.clone()
    }

    // -- Data extraction --

    pub fn int64_value(&self, indices: &[i64]) -> i64 {
        if indices.is_empty() {
            return self.inner.item_i64();
        }
        let starts: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        let stops: Vec<i32> = indices.iter().map(|&i| i as i32 + 1).collect();
        let strides: Vec<i32> = vec![1; indices.len()];
        let sliced = crate::backend::mlx::ops::slice(&self.inner, &starts, &stops, &strides);
        sliced.item_i64()
    }

    pub fn f64_value(&self, indices: &[i64]) -> f64 {
        if indices.is_empty() {
            return self.inner.item_f32() as f64;
        }
        let starts: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        let stops: Vec<i32> = indices.iter().map(|&i| i as i32 + 1).collect();
        let strides: Vec<i32> = vec![1; indices.len()];
        let sliced = crate::backend::mlx::ops::slice(&self.inner, &starts, &stops, &strides);
        sliced.item_f32() as f64
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        let f32_arr = self.inner.astype(crate::backend::mlx::ffi::mlx_dtype::MLX_FLOAT32);
        f32_arr.to_vec_f32()
    }
}

// ---------------------------------------------------------------------------
// Operator overloads (both backends)
// ---------------------------------------------------------------------------

// Add: Tensor + Tensor
impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner + &rhs.inner) }
        #[cfg(feature = "mlx")]
        { Tensor::from_mlx(crate::backend::mlx::ops::add(&self.inner, &rhs.inner)) }
    }
}

impl std::ops::Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor { self + &rhs }
}

impl std::ops::Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor { &self + rhs }
}

impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor { &self + &rhs }
}

// Add: Tensor + f64
impl std::ops::Add<f64> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner + rhs) }
        #[cfg(feature = "mlx")]
        {
            let scalar = crate::backend::mlx::array::MlxArray::scalar_f32(rhs as f32);
            Tensor::from_mlx(crate::backend::mlx::ops::add(&self.inner, &scalar))
        }
    }
}

impl std::ops::Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor { &self + rhs }
}

// Sub: Tensor - Tensor
impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner - &rhs.inner) }
        #[cfg(feature = "mlx")]
        { Tensor::from_mlx(crate::backend::mlx::ops::subtract(&self.inner, &rhs.inner)) }
    }
}

impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor { self - &rhs }
}

impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor { &self - rhs }
}

impl std::ops::Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor { &self - &rhs }
}

impl std::ops::Sub<f64> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f64) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner - rhs) }
        #[cfg(feature = "mlx")]
        {
            let scalar = crate::backend::mlx::array::MlxArray::scalar_f32(rhs as f32);
            Tensor::from_mlx(crate::backend::mlx::ops::subtract(&self.inner, &scalar))
        }
    }
}

// Mul: Tensor * Tensor
impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner * &rhs.inner) }
        #[cfg(feature = "mlx")]
        { Tensor::from_mlx(crate::backend::mlx::ops::multiply(&self.inner, &rhs.inner)) }
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor { self * &rhs }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor { &self * rhs }
}

impl std::ops::Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor { &self * &rhs }
}

// Mul: Tensor * f64
impl std::ops::Mul<f64> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner * rhs) }
        #[cfg(feature = "mlx")]
        {
            let scalar = crate::backend::mlx::array::MlxArray::scalar_f32(rhs as f32);
            Tensor::from_mlx(crate::backend::mlx::ops::multiply(&self.inner, &scalar))
        }
    }
}

impl std::ops::Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor { &self * rhs }
}

// Div: Tensor / Tensor
impl std::ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner / &rhs.inner) }
        #[cfg(feature = "mlx")]
        { Tensor::from_mlx(crate::backend::mlx::ops::divide(&self.inner, &rhs.inner)) }
    }
}

impl std::ops::Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor { self / &rhs }
}

impl std::ops::Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor { &self / rhs }
}

impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor { &self / &rhs }
}

// Div: Tensor / f64
impl std::ops::Div<f64> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner / rhs) }
        #[cfg(feature = "mlx")]
        {
            let scalar = crate::backend::mlx::array::MlxArray::scalar_f32(rhs as f32);
            Tensor::from_mlx(crate::backend::mlx::ops::divide(&self.inner, &scalar))
        }
    }
}

impl std::ops::Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor { &self / rhs }
}

// Neg: -Tensor
impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(-&self.inner) }
        #[cfg(feature = "mlx")]
        { Tensor::from_mlx(crate::backend::mlx::ops::negative(&self.inner)) }
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor { -&self }
}

// AddAssign
impl std::ops::AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        *self = &*self + rhs;
    }
}

impl std::ops::AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, rhs: Tensor) {
        *self = &*self + &rhs;
    }
}
