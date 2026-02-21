//! GGUF (GGML Universal File Format) loader.
//!
//! Parses GGUF v3 files which contain model metadata and quantized weights.
//! Supports dequantization of Q8_0 and Q4_K_M formats to f16 for inference.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use forge_core::{Backend, ForgeError, Result};
use memmap2::Mmap;

use crate::gguf_dequant;

const GGUF_MAGIC: u32 = 0x46475547; // "GGUF" in little-endian

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// GGUF quantization types we recognize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            9 => Ok(GgmlType::Q8_1),
            10 => Ok(GgmlType::Q2K),
            11 => Ok(GgmlType::Q3K),
            12 => Ok(GgmlType::Q4K),
            13 => Ok(GgmlType::Q5K),
            14 => Ok(GgmlType::Q6K),
            _ => Err(ForgeError::ModelLoad(format!(
                "Unknown GGML type: {v}"
            ))),
        }
    }

    /// Block size in elements for this quantization type.
    pub fn block_size(&self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 => 32,
            GgmlType::Q5_0 | GgmlType::Q5_1 => 32,
            GgmlType::Q8_0 | GgmlType::Q8_1 => 32,
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => 256,
        }
    }

    /// Bytes per block for this quantization type.
    pub fn type_size(&self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 18,    // 2 (scale) + 16 (4-bit × 32 / 8)
            GgmlType::Q4_1 => 20,    // 2 (scale) + 2 (min) + 16
            GgmlType::Q5_0 => 22,    // 2 + 4 (high bits) + 16
            GgmlType::Q5_1 => 24,    // 2 + 2 + 4 + 16
            GgmlType::Q8_0 => 34,    // 2 (f16 scale) + 32 (i8 × 32)
            GgmlType::Q8_1 => 40,    // 4 (f32 scale) + 4 (f32 min) + 32
            GgmlType::Q2K => 256 / 16 * 2 + 256 / 4 + 2 + 2, // simplified
            GgmlType::Q3K => 256 / 8 * 3 + 256 / 4 + 2,      // simplified
            GgmlType::Q4K => 144,    // 2+2+12+128 (Q4_K block for 256 elements)
            GgmlType::Q5K => 176,    // similar
            GgmlType::Q6K => 210,    // similar
        }
    }
}

/// Info about a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset from the start of the tensor data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Total bytes this tensor occupies in the file.
    pub fn byte_size(&self) -> usize {
        let n_elements = self.num_elements();
        let block_size = self.ggml_type.block_size();
        let n_blocks = (n_elements + block_size - 1) / block_size;
        n_blocks * self.ggml_type.type_size()
    }
}

/// Loader for GGUF format model files.
pub struct GgufLoader {
    mmap: Mmap,
    /// GGUF file version.
    pub version: u32,
    /// Metadata key-value pairs.
    pub metadata: HashMap<String, GgufValue>,
    /// Tensor information indexed by name.
    tensor_infos: HashMap<String, GgufTensorInfo>,
    /// Byte offset where tensor data begins.
    data_offset: u64,
}

impl GgufLoader {
    /// Open and parse a GGUF file.
    pub fn new(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let mut cursor = Cursor::new(&mmap[..]);

        // Header
        let magic = read_u32(&mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(ForgeError::ModelLoad(format!(
                "Not a GGUF file (magic: 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x})"
            )));
        }

        let version = read_u32(&mut cursor)?;
        if version < 2 || version > 3 {
            return Err(ForgeError::ModelLoad(format!(
                "Unsupported GGUF version: {version} (only v2 and v3 supported)"
            )));
        }

        let n_tensors = read_u64(&mut cursor)? as usize;
        let n_kv = read_u64(&mut cursor)? as usize;

        // Parse metadata
        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = read_gguf_string(&mut cursor)?;
            let value = read_gguf_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Parse tensor infos
        let mut tensor_infos = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = read_u32(&mut cursor)? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u64(&mut cursor)?);
            }
            let ggml_type = GgmlType::from_u32(read_u32(&mut cursor)?)?;
            let offset = read_u64(&mut cursor)?;

            tensor_infos.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    shape,
                    ggml_type,
                    offset,
                },
            );
        }

        // Data section starts at an aligned position after the header.
        // GGUF aligns tensor data to 32 bytes (or to the alignment specified in metadata).
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) if *a > 0 => *a as u64,
            Some(GgufValue::U64(a)) if *a > 0 => *a,
            Some(_) => {
                return Err(ForgeError::ModelLoad(
                    "GGUF general.alignment must be > 0".into(),
                ));
            }
            None => 32,
        };
        let header_end = cursor.position();
        let data_offset = (header_end + alignment - 1) / alignment * alignment;

        Ok(Self {
            mmap,
            version,
            metadata,
            tensor_infos,
            data_offset,
        })
    }

    /// Load a tensor by name, dequantizing to f16 and uploading to the backend.
    pub fn load_tensor<B: Backend>(&self, name: &str, backend: &B) -> Result<B::Tensor> {
        let info = self.tensor_infos.get(name).ok_or_else(|| {
            ForgeError::ModelLoad(format!("Tensor '{name}' not found in GGUF file"))
        })?;

        let data_start = (self.data_offset + info.offset) as usize;
        let data_end = data_start + info.byte_size();
        if data_end > self.mmap.len() {
            return Err(ForgeError::ModelLoad(format!(
                "Tensor '{name}' data extends past end of file"
            )));
        }
        let raw_data = &self.mmap[data_start..data_end];
        let n_elements = info.num_elements();
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();

        match info.ggml_type {
            GgmlType::F32 => {
                let f32_data: &[f32] = bytemuck::try_cast_slice(raw_data)
                    .map_err(|e| ForgeError::ModelLoad(format!("F32 alignment error: {e}")))?;
                backend.copy_from_host_f32(f32_data, &shape)
            }
            GgmlType::F16 => {
                let f16_data: &[half::f16] = bytemuck::try_cast_slice(raw_data)
                    .map_err(|e| ForgeError::ModelLoad(format!("F16 alignment error: {e}")))?;
                backend.copy_from_host_f16(f16_data, &shape)
            }
            GgmlType::Q8_0 => {
                let f16_data = gguf_dequant::dequantize_q8_0(raw_data, n_elements)?;
                backend.copy_from_host_f16(&f16_data, &shape)
            }
            GgmlType::Q4K => {
                let f16_data = gguf_dequant::dequantize_q4_k(raw_data, n_elements)?;
                backend.copy_from_host_f16(&f16_data, &shape)
            }
            other => Err(ForgeError::ModelLoad(format!(
                "Unsupported GGML quantization type for tensor '{name}': {other:?}. \
                 Only F32, F16, Q8_0, and Q4_K are currently supported."
            ))),
        }
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensor_infos.keys().cloned().collect()
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_infos.get(name)
    }

    /// Get a metadata string value.
    pub fn get_metadata_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Get a metadata u32 value.
    pub fn get_metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufValue::U32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a metadata u64 value.
    pub fn get_metadata_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(GgufValue::U64(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a metadata f32 value.
    pub fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufValue::F32(v)) => Some(*v),
            _ => None,
        }
    }
}

// --- GGUF binary format readers ---

fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut buf = [0u8; 1];
    cursor.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(cursor: &mut Cursor<&[u8]>) -> Result<i8> {
    Ok(read_u8(cursor)? as i8)
}

fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(cursor: &mut Cursor<&[u8]>) -> Result<i16> {
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(cursor: &mut Cursor<&[u8]>) -> Result<i64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(cursor: &mut Cursor<&[u8]>) -> Result<f64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bool(cursor: &mut Cursor<&[u8]>) -> Result<bool> {
    Ok(read_u8(cursor)? != 0)
}

fn read_gguf_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = read_u64(cursor)? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| ForgeError::ModelLoad(format!("Invalid UTF-8 in GGUF string: {e}")))
}

/// GGUF value type IDs
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;

fn read_gguf_value(cursor: &mut Cursor<&[u8]>) -> Result<GgufValue> {
    let type_id = read_u32(cursor)?;
    read_gguf_typed_value(cursor, type_id)
}

fn read_gguf_typed_value(cursor: &mut Cursor<&[u8]>, type_id: u32) -> Result<GgufValue> {
    match type_id {
        GGUF_TYPE_U8 => Ok(GgufValue::U8(read_u8(cursor)?)),
        GGUF_TYPE_I8 => Ok(GgufValue::I8(read_i8(cursor)?)),
        GGUF_TYPE_U16 => Ok(GgufValue::U16(read_u16(cursor)?)),
        GGUF_TYPE_I16 => Ok(GgufValue::I16(read_i16(cursor)?)),
        GGUF_TYPE_U32 => Ok(GgufValue::U32(read_u32(cursor)?)),
        GGUF_TYPE_I32 => Ok(GgufValue::I32(read_i32(cursor)?)),
        GGUF_TYPE_F32 => Ok(GgufValue::F32(read_f32(cursor)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_bool(cursor)?)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_gguf_string(cursor)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(cursor)?;
            let len = read_u64(cursor)? as usize;
            const MAX_ARRAY_LEN: usize = 10_000_000;
            if len > MAX_ARRAY_LEN {
                return Err(ForgeError::ModelLoad(format!(
                    "GGUF array length {len} exceeds maximum {MAX_ARRAY_LEN}"
                )));
            }
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_gguf_typed_value(cursor, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        GGUF_TYPE_U64 => Ok(GgufValue::U64(read_u64(cursor)?)),
        GGUF_TYPE_I64 => Ok(GgufValue::I64(read_i64(cursor)?)),
        GGUF_TYPE_F64 => Ok(GgufValue::F64(read_f64(cursor)?)),
        _ => Err(ForgeError::ModelLoad(format!(
            "Unknown GGUF value type: {type_id}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GGUF v3 file in memory.
    fn build_test_gguf(
        metadata: &[(&str, GgufValue)],
        tensors: &[(&str, &[u64], GgmlType, &[u8])],
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version
        buf.extend_from_slice(&3u32.to_le_bytes());
        // n_tensors
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // n_kv
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        // Metadata KV pairs
        for (key, value) in metadata {
            write_gguf_string(&mut buf, key);
            write_gguf_value(&mut buf, value);
        }

        // Tensor infos — compute offsets
        let alignment = 32u64;
        let mut tensor_data_sections: Vec<&[u8]> = Vec::new();
        let mut current_offset = 0u64;

        for (name, shape, ggml_type, data) in tensors {
            write_gguf_string(&mut buf, name);
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in *shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&(*ggml_type as u32).to_le_bytes());
            buf.extend_from_slice(&current_offset.to_le_bytes());
            current_offset += data.len() as u64;
            tensor_data_sections.push(data);
        }

        // Align to 32 bytes
        let header_end = buf.len() as u64;
        let data_start = (header_end + alignment - 1) / alignment * alignment;
        buf.resize(data_start as usize, 0);

        // Tensor data
        for data in tensor_data_sections {
            buf.extend_from_slice(data);
        }

        buf
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_gguf_value(buf: &mut Vec<u8>, value: &GgufValue) {
        match value {
            GgufValue::U32(v) => {
                buf.extend_from_slice(&GGUF_TYPE_U32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::String(s) => {
                buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
                write_gguf_string(buf, s);
            }
            GgufValue::F32(v) => {
                buf.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            _ => unimplemented!("test helper only supports U32, String, F32"),
        }
    }

    #[test]
    fn test_parse_gguf_header() {
        let data = build_test_gguf(
            &[
                ("general.architecture", GgufValue::String("llama".into())),
                ("llama.context_length", GgufValue::U32(4096)),
            ],
            &[], // no tensors
        );

        // Write to temp file
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();

        let loader = GgufLoader::new(tmp.path()).unwrap();
        assert_eq!(loader.version, 3);
        assert_eq!(
            loader.get_metadata_string("general.architecture"),
            Some("llama")
        );
        assert_eq!(loader.get_metadata_u32("llama.context_length"), Some(4096));
    }

    #[test]
    fn test_parse_gguf_tensors() {
        // Build a file with one F32 tensor [2, 3] = 6 elements = 24 bytes
        let tensor_data = vec![0u8; 24];
        let data = build_test_gguf(
            &[],
            &[("model.embed_tokens.weight", &[2, 3], GgmlType::F32, &tensor_data)],
        );

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();

        let loader = GgufLoader::new(tmp.path()).unwrap();
        let names = loader.tensor_names();
        assert_eq!(names.len(), 1);

        let info = loader.tensor_info("model.embed_tokens.weight").unwrap();
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.ggml_type, GgmlType::F32);
        assert_eq!(info.num_elements(), 6);
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = vec![0u8; 100];
        // Wrong magic
        data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();

        assert!(GgufLoader::new(tmp.path()).is_err());
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&99u32.to_le_bytes()); // unsupported version
        data.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        data.extend_from_slice(&0u64.to_le_bytes()); // n_kv

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();

        assert!(GgufLoader::new(tmp.path()).is_err());
    }
}
