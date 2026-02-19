use std::path::Path;

use forge_core::{Backend, ForgeError, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

pub struct SafeTensorsLoader {
    mmaps: Vec<Mmap>,
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
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { mmaps })
    }

    /// Load a specific tensor by name.
    pub fn load_tensor<B: Backend>(&self, name: &str, backend: &B) -> Result<B::Tensor> {
        for mmap in &self.mmaps {
            let tensors = SafeTensors::deserialize(mmap)
                .map_err(|e| ForgeError::ModelLoad(e.to_string()))?;
            if let Ok(view) = tensors.tensor(name) {
                let shape: Vec<usize> = view.shape().to_vec();
                return view_to_tensor(view, &shape, backend);
            }
        }
        Err(ForgeError::ModelLoad(format!(
            "Tensor '{}' not found",
            name
        )))
    }

    /// List all tensor names across all files.
    pub fn tensor_names(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for mmap in &self.mmaps {
            let tensors = SafeTensors::deserialize(mmap)
                .map_err(|e| ForgeError::ModelLoad(e.to_string()))?;
            names.extend(tensors.names().into_iter().map(String::from));
        }
        Ok(names)
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
            let f16_data: &[half::f16] = bytemuck::cast_slice(data);
            backend.copy_from_host_f16(f16_data, shape)
        }
        safetensors::Dtype::BF16 => {
            let bf16_data: &[half::bf16] = bytemuck::cast_slice(data);
            backend.copy_from_host_bf16(bf16_data, shape)
        }
        safetensors::Dtype::F32 => {
            let f32_data: &[f32] = bytemuck::cast_slice(data);
            backend.copy_from_host_f32(f32_data, shape)
        }
        other => Err(ForgeError::ModelLoad(format!(
            "Unsupported safetensors dtype: {:?}",
            other
        ))),
    }
}
