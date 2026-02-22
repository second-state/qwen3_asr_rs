//! MLX safetensors I/O.

use super::array::MlxArray;
use super::ffi;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

/// Load all tensors from a safetensors file.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, MlxArray>, String> {
    let path_str = path.to_str().ok_or("Invalid UTF-8 in path")?;
    let c_path = CString::new(path_str).map_err(|e| format!("Invalid path: {e}"))?;

    let mut data: ffi::mlx_map_string_to_array = std::ptr::null_mut();
    let mut metadata: ffi::mlx_map_string_to_string = std::ptr::null_mut();

    // Use a CPU stream for loading: Load::eval_gpu is not implemented for Metal.
    let cpu_device = unsafe { ffi::mlx_device_new_type(ffi::mlx_device_type::MLX_CPU, 0) };
    let cpu_stream = unsafe { ffi::mlx_stream_new_device(cpu_device) };
    unsafe { ffi::mlx_device_free(cpu_device) };

    let status = unsafe {
        ffi::mlx_load_safetensors(&mut data, &mut metadata, c_path.as_ptr(), cpu_stream)
    };
    unsafe { ffi::mlx_stream_free(cpu_stream) };

    if status != 0 {
        return Err(format!("Failed to load safetensors from {:?}", path));
    }

    let mut result = HashMap::new();

    let iter = unsafe { ffi::mlx_map_string_to_array_iterator_new(data) };
    if iter.is_null() {
        unsafe { ffi::mlx_map_string_to_array_free(data) };
        if !metadata.is_null() {
            unsafe { ffi::mlx_map_string_to_string_free(metadata) };
        }
        return Err("Failed to create map iterator".to_string());
    }

    loop {
        let mut key_ptr: *const std::os::raw::c_char = std::ptr::null();
        let mut arr = MlxArray::empty();
        let status = unsafe {
            ffi::mlx_map_string_to_array_iterator_next(&mut key_ptr, &mut arr.ptr, iter)
        };
        if status != 0 || key_ptr.is_null() {
            break;
        }

        let name = unsafe { std::ffi::CStr::from_ptr(key_ptr) }
            .to_string_lossy()
            .into_owned();

        result.insert(name, arr);
    }

    unsafe { ffi::mlx_map_string_to_array_iterator_free(iter) };
    unsafe { ffi::mlx_map_string_to_array_free(data) };
    if !metadata.is_null() {
        unsafe { ffi::mlx_map_string_to_string_free(metadata) };
    }

    Ok(result)
}

/// Load all tensors from multiple safetensors shards in a directory.
pub fn load_safetensors_dir(dir: &Path) -> Result<HashMap<String, MlxArray>, String> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return load_safetensors(&single);
    }

    let pattern_prefix = "model-";
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {:?}: {e}", dir))?;

    let mut shard_files: Vec<_> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().into_string().ok()?;
            if name.starts_with(pattern_prefix) && name.ends_with(".safetensors") {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();

    shard_files.sort();

    if shard_files.is_empty() {
        return Err(format!("No safetensors files found in {:?}", dir));
    }

    let mut all_tensors = HashMap::new();
    for shard_path in &shard_files {
        let shard_tensors = load_safetensors(shard_path)?;
        all_tensors.extend(shard_tensors);
    }

    Ok(all_tensors)
}
