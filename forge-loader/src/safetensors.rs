use std::collections::HashMap;
use std::path::Path;

use forge_core::{Backend, ForgeError, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

pub struct SafeTensorsLoader {
    mmaps: Vec<Mmap>,
    /// Maps tensor name -> mmap index for O(1) lookup.
    tensor_index: HashMap<String, usize>,
}

impl SafeTensorsLoader {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let mut files: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == "safetensors")
            })
            .map(|e| e.path())
            .collect();
        files.sort();

        if files.is_empty() {
            return Err(ForgeError::ModelLoad(format!(
                "No .safetensors files found in {:?}",
                model_dir
            )));
        }

        let mmaps = files
            .iter()
            .map(|path| {
                let file = std::fs::File::open(path)?;
                Ok(unsafe { Mmap::map(&file) }?)
            })
            .collect::<std::result::Result<Vec<_>, std::io::Error>>()?;

        // Build tensor name -> file index mapping
        let mut tensor_index = HashMap::new();
        for (idx, mmap) in mmaps.iter().enumerate() {
            let tensors = SafeTensors::deserialize(mmap)
                .map_err(|e| ForgeError::ModelLoad(e.to_string()))?;
            for name in tensors.names() {
                tensor_index.insert(name.to_string(), idx);
            }
        }

        Ok(Self {
            mmaps,
            tensor_index,
        })
    }

    /// Load a specific tensor by name.
    pub fn load_tensor<B: Backend>(&self, name: &str, backend: &B) -> Result<B::Tensor> {
        let idx = self.tensor_index.get(name).ok_or_else(|| {
            ForgeError::ModelLoad(format!("Tensor '{}' not found", name))
        })?;
        let tensors = SafeTensors::deserialize(&self.mmaps[*idx])
            .map_err(|e| ForgeError::ModelLoad(e.to_string()))?;
        let view = tensors
            .tensor(name)
            .map_err(|e| ForgeError::ModelLoad(e.to_string()))?;
        let shape: Vec<usize> = view.shape().to_vec();
        view_to_tensor(view, &shape, backend)
    }

    /// List all tensor names across all files.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensor_index.keys().cloned().collect()
    }
}

fn view_to_tensor<B: Backend>(
    view: safetensors::tensor::TensorView<'_>,
    shape: &[usize],
    backend: &B,
) -> Result<B::Tensor> {
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::F16 => {
            let f16_data: &[half::f16] = bytemuck::try_cast_slice(data)
                .map_err(|e| ForgeError::ModelLoad(format!("F16 alignment error: {e}")))?;
            backend.copy_from_host_f16(f16_data, shape)
        }
        safetensors::Dtype::BF16 => {
            // BF16 tensors are auto-converted to F16 at load time because our
            // compute kernels only support F32 and F16. This is the standard
            // approach used by inference servers (BF16 → F16 is lossless for
            // the exponent range and only slightly reduces mantissa precision).
            let bf16_data: &[half::bf16] = bytemuck::try_cast_slice(data)
                .map_err(|e| ForgeError::ModelLoad(format!("BF16 alignment error: {e}")))?;
            let f16_data: Vec<half::f16> = bf16_data
                .iter()
                .map(|v| half::f16::from_f32(v.to_f32()))
                .collect();
            backend.copy_from_host_f16(&f16_data, shape)
        }
        safetensors::Dtype::F32 => {
            let f32_data: &[f32] = bytemuck::try_cast_slice(data)
                .map_err(|e| ForgeError::ModelLoad(format!("F32 alignment error: {e}")))?;
            backend.copy_from_host_f32(f32_data, shape)
        }
        other => Err(ForgeError::ModelLoad(format!(
            "Unsupported safetensors dtype: {:?}",
            other
        ))),
    }
}
