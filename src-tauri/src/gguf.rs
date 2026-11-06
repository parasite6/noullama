//! Just enough of the GGUF format to tell how a file relates to multi-token
//! prediction (MTP), from its metadata rather than its name:
//!
//! - a model with built-in MTP layers has `<arch>.nextn_predict_layers` set
//!   to fewer layers than its `<arch>.block_count`;
//! - a separate MTP head is a small model made only of those layers
//!   (`nextn_predict_layers` equals `block_count`). Its output width,
//!   `<arch>.embedding_length_out`, must match the main model's
//!   `<arch>.embedding_length` for llama.cpp to pair them.
//!
//! It also keeps the architecture, the model's name and whether the file
//! declares native decision support (`<arch>.decision.type`), which decide
//! how decisions are read from it (see `decision.rs`).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use std::sync::Mutex;
use std::time::SystemTime;

const MAGIC: &[u8; 4] = b"GGUF";
// Generous ceilings so a damaged or hostile file cannot make the reader
// allocate or loop without bound; real models stay far below them.
const MAX_KV: u64 = 1 << 16;
const MAX_KEY: u64 = 1 << 16;
const MAX_STRING: u64 = 1 << 30;
const MAX_ARRAY: u64 = 1 << 26;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Info {
    pub architecture: String,
    /// `general.name`, then `general.basename`, joined by a space.
    pub name: String,
    /// The file has `<arch>.decision.type`, so llama.cpp serves it natively.
    pub native_decisions: bool,
    pub block_count: u64,
    pub nextn_layers: u64,
    pub embedding_length: u64,
    pub embedding_length_out: u64,
}

impl Info {
    /// A model whose own weights include MTP layers.
    pub fn has_built_in_mtp(&self) -> bool {
        self.nextn_layers > 0 && self.nextn_layers < self.block_count
    }
    /// A standalone MTP head, loaded next to a model as a draft.
    pub fn is_mtp_head(&self) -> bool {
        self.nextn_layers > 0 && self.nextn_layers >= self.block_count
    }
    /// Whether this head can draft for `model`: its output feeds the model's
    /// hidden state, so the widths must agree. A width either file does not
    /// record is not held against the pair.
    pub fn head_fits(&self, model: &Info) -> bool {
        let out = if self.embedding_length_out > 0 { self.embedding_length_out } else { self.embedding_length };
        out == 0 || model.embedding_length == 0 || out == model.embedding_length
    }
}

type Key = (String, u64, Option<SystemTime>);
static CACHE: Mutex<Option<HashMap<Key, Option<Info>>>> = Mutex::new(None);

/// The MTP-related metadata of a GGUF file, or `None` when it is not a
/// readable GGUF. Results are cached by path, size and modification time, so
/// repeated scans do not reread the file.
pub fn info(path: &str) -> Option<Info> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() {
        return None;
    }
    let key = (path.to_string(), meta.len(), meta.modified().ok());
    if let Some(found) = CACHE.lock().ok().and_then(|c| c.as_ref()?.get(&key).cloned()) {
        return found;
    }
    let found = File::open(path)
        .ok()
        .and_then(|file| read_info(&mut BufReader::with_capacity(1 << 16, file)));
    if let Ok(mut cache) = CACHE.lock() {
        cache.get_or_insert_with(HashMap::new).insert(key, found.clone());
    }
    found
}

/// Whether a local file is a standalone MTP head.
pub fn is_mtp_head(path: &str) -> bool {
    info(path).is_some_and(|i| i.is_mtp_head())
}

/// Whether a file name in a Hugging Face listing looks like an MTP head.
/// Remote files cannot be read, so this goes by the name: "mtp" as its own
/// word, as in llama.cpp's `mtp-` sidecars or `model-mtp.gguf`.
pub fn named_like_mtp_head(name: &str) -> bool {
    let name = name.rsplit('/').next().unwrap_or(name).to_lowercase();
    name.ends_with(".gguf")
        && name
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|word| word == "mtp")
}

/// How a model can use MTP: through its own layers, or through a head in
/// the same folder whose width fits it.
#[derive(Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ModelMtp {
    pub built_in: bool,
    pub head: Option<String>,
}

pub fn model_mtp(path: &str) -> ModelMtp {
    let Some(model) = info(path) else { return ModelMtp::default() };
    if model.is_mtp_head() {
        return ModelMtp::default();
    }
    let head = Path::new(path).parent().and_then(|dir| {
        let mut heads: Vec<String> = std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|p| p != path && p.to_lowercase().ends_with(".gguf"))
            .filter(|p| info(p).is_some_and(|h| h.is_mtp_head() && h.head_fits(&model)))
            .collect();
        heads.sort();
        heads.into_iter().next()
    });
    ModelMtp { built_in: model.has_built_in_mtp(), head }
}

fn read_info<R: Read + Seek>(r: &mut R) -> Option<Info> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).ok()?;
    if &magic != MAGIC {
        return None;
    }
    // Version 1 used 32-bit lengths and predates MTP support.
    if !matches!(u32le(r)?, 2 | 3) {
        return None;
    }
    let _tensors = u64le(r)?;
    let kv_count = u64le(r)?;
    if kv_count > MAX_KV {
        return None;
    }
    // Keys are prefixed with the architecture, which other metadata (a
    // projector's `clip.vision.block_count`) must not be mistaken for, so
    // collect candidates and resolve them once the architecture is known.
    let mut arch = String::new();
    let mut names: Vec<(String, String)> = Vec::new();
    let mut decision_keys: Vec<String> = Vec::new();
    let mut numbers: HashMap<String, u64> = HashMap::new();
    for _ in 0..kv_count {
        let len = u64le(r)?;
        if len > MAX_KEY {
            return None;
        }
        let mut key = vec![0u8; len as usize];
        r.read_exact(&mut key).ok()?;
        let key = String::from_utf8_lossy(&key).into_owned();
        let kind = u32le(r)?;
        if kind == 8 && ["general.architecture", "general.name", "general.basename"].contains(&key.as_str()) {
            let len = u64le(r)?;
            if len > MAX_KEY {
                return None;
            }
            let mut value = vec![0u8; len as usize];
            r.read_exact(&mut value).ok()?;
            let value = String::from_utf8_lossy(&value).into_owned();
            if key == "general.architecture" {
                arch = value;
            } else {
                names.push((key, value));
            }
        } else if key.ends_with(".decision.type") {
            decision_keys.push(key);
            skip(r, kind)?;
        } else if [".block_count", ".nextn_predict_layers", ".embedding_length", ".embedding_length_out"]
            .iter()
            .any(|suffix| key.ends_with(suffix))
        {
            if fixed_size(kind).is_some() && kind != 6 && kind != 7 && kind != 12 {
                let n = integer(r, kind)?;
                numbers.insert(key, n.clamp(0, u64::MAX as i128) as u64);
            } else {
                skip(r, kind)?;
            }
        } else {
            skip(r, kind)?;
        }
    }
    // Skipping only seeks, and seeking past the end does not fail, so check
    // that the metadata ended inside the file.
    let end = r.stream_position().ok()?;
    if end > r.seek(std::io::SeekFrom::End(0)).ok()? {
        return None;
    }
    let get = |name: &str| numbers.get(&format!("{arch}.{name}")).copied().unwrap_or(0);
    names.sort_by_key(|(key, _)| key != "general.name");
    Some(Info {
        native_decisions: decision_keys.contains(&format!("{arch}.decision.type")),
        name: names.into_iter().map(|(_, v)| v).collect::<Vec<_>>().join(" "),
        architecture: arch.clone(),
        block_count: get("block_count"),
        nextn_layers: get("nextn_predict_layers"),
        embedding_length: get("embedding_length"),
        embedding_length_out: get("embedding_length_out"),
    })
}

fn integer<R: Read>(r: &mut R, kind: u32) -> Option<i128> {
    Some(match kind {
        0 => u8le(r)? as i128,
        1 => u8le(r)? as i8 as i128,
        2 => u16::from_le_bytes(bytes(r)?) as i128,
        3 => i16::from_le_bytes(bytes(r)?) as i128,
        4 => u32le(r)? as i128,
        5 => i32::from_le_bytes(bytes(r)?) as i128,
        10 => u64le(r)? as i128,
        11 => i64::from_le_bytes(bytes(r)?) as i128,
        _ => return None,
    })
}

fn fixed_size(kind: u32) -> Option<u64> {
    Some(match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        _ => return None,
    })
}

fn skip<R: Read + Seek>(r: &mut R, kind: u32) -> Option<()> {
    if let Some(size) = fixed_size(kind) {
        return seek(r, size);
    }
    match kind {
        8 => {
            let len = u64le(r)?;
            if len > MAX_STRING {
                return None;
            }
            seek(r, len)
        }
        9 => {
            let inner = u32le(r)?;
            let count = u64le(r)?;
            if count > MAX_ARRAY {
                return None;
            }
            if let Some(size) = fixed_size(inner) {
                return seek(r, size.checked_mul(count)?);
            }
            // Arrays of strings (tokenizer vocabularies) must be walked; nested
            // arrays are not used by llama.cpp and are treated as unreadable.
            if inner != 8 {
                return None;
            }
            for _ in 0..count {
                skip(r, 8)?;
            }
            Some(())
        }
        _ => None,
    }
}

fn seek<R: Seek>(r: &mut R, n: u64) -> Option<()> {
    r.seek_relative(i64::try_from(n).ok()?).ok()
}
fn bytes<const N: usize, R: Read>(r: &mut R) -> Option<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b).ok()?;
    Some(b)
}
fn u8le<R: Read>(r: &mut R) -> Option<u8> {
    bytes::<1, R>(r).map(|b| b[0])
}
fn u32le<R: Read>(r: &mut R) -> Option<u32> {
    bytes(r).map(u32::from_le_bytes)
}
fn u64le<R: Read>(r: &mut R) -> Option<u64> {
    bytes(r).map(u64::from_le_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    fn kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
        string(out, key);
        out.extend(4u32.to_le_bytes());
        out.extend(value.to_le_bytes());
    }

    /// A GGUF header for `arch` with a string array and a float array before
    /// the numbers, and a projector-style key that must be ignored.
    fn gguf(arch: &str, numbers: &[(&str, u32)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend((4 + numbers.len() as u64).to_le_bytes());
        string(&mut out, "general.architecture");
        out.extend(8u32.to_le_bytes());
        string(&mut out, arch);
        string(&mut out, "tokenizer.ggml.tokens");
        out.extend(9u32.to_le_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        string(&mut out, "<s>");
        string(&mut out, "hello");
        string(&mut out, "tokenizer.ggml.scores");
        out.extend(9u32.to_le_bytes());
        out.extend(6u32.to_le_bytes());
        out.extend(3u64.to_le_bytes());
        out.extend([0u8; 12]);
        kv_u32(&mut out, "clip.vision.block_count", 99);
        for (name, value) in numbers {
            kv_u32(&mut out, &format!("{arch}.{name}"), *value);
        }
        out
    }
    fn parse(data: Vec<u8>) -> Option<Info> {
        read_info(&mut Cursor::new(data))
    }

    #[test]
    fn tells_built_in_layers_from_a_separate_head() {
        let built_in = parse(gguf("qwen35", &[("block_count", 41), ("nextn_predict_layers", 1)])).unwrap();
        assert!(built_in.has_built_in_mtp() && !built_in.is_mtp_head());
        // The shape of winnow-mtp.gguf: four blocks, all of them MTP layers.
        let head = parse(gguf("gemma4-assistant", &[("block_count", 4), ("nextn_predict_layers", 4), ("embedding_length", 256), ("embedding_length_out", 2560)])).unwrap();
        assert!(head.is_mtp_head() && !head.has_built_in_mtp());
        let plain = parse(gguf("gemma4", &[("block_count", 42), ("embedding_length", 2560)])).unwrap();
        assert!(!plain.has_built_in_mtp() && !plain.is_mtp_head());
        assert_eq!(plain.block_count, 42, "the projector's block_count must not leak in");
        assert!(head.head_fits(&plain));
        let other = parse(gguf("llama", &[("block_count", 32), ("embedding_length", 4096)])).unwrap();
        assert!(!head.head_fits(&other));
    }

    #[test]
    fn reads_the_name_and_native_decision_key() {
        let extra = |data: &mut Vec<u8>, key: &str, value: &str| {
            string(data, key);
            data.extend(8u32.to_le_bytes());
            string(data, value);
        };
        // The kv count sits after the magic, version and tensor count.
        let set_count = |data: &mut Vec<u8>, n: u64| data[16..24].copy_from_slice(&n.to_le_bytes());
        let mut data = gguf("gemma4", &[("block_count", 42)]);
        extra(&mut data, "general.basename", "Winnow");
        extra(&mut data, "general.name", "Winnow Mini E4B");
        set_count(&mut data, 7);
        let info = parse(data.clone()).unwrap();
        assert_eq!(info.architecture, "gemma4");
        assert_eq!(info.name, "Winnow Mini E4B Winnow");
        assert!(!info.native_decisions);
        extra(&mut data, "gemma4.decision.type", "systemone");
        set_count(&mut data, 8);
        assert!(parse(data).unwrap().native_decisions);
        // Another architecture's key does not count.
        let mut other = gguf("llama", &[]);
        extra(&mut other, "gemma4.decision.type", "systemone");
        set_count(&mut other, 5);
        assert!(!parse(other).unwrap().native_decisions);
    }

    #[test]
    fn rejects_bad_magic_truncation_and_huge_counts() {
        assert_eq!(parse(b"GGML\x03\0\0\0".to_vec()), None);
        let mut data = gguf("qwen35", &[("nextn_predict_layers", 1)]);
        data.truncate(data.len() - 2);
        assert_eq!(parse(data), None);
        let mut huge = b"GGUF".to_vec();
        huge.extend(3u32.to_le_bytes());
        huge.extend(0u64.to_le_bytes());
        huge.extend(u64::MAX.to_le_bytes());
        assert_eq!(parse(huge), None);
    }

    #[test]
    fn names_remote_heads_by_the_word_mtp() {
        assert!(named_like_mtp_head("mtp-Qwen3.5-9B-Q8_0.gguf"));
        assert!(named_like_mtp_head("gguf/winnow-mtp.gguf"));
        assert!(named_like_mtp_head("Model-MTP-F16.gguf"));
        assert!(!named_like_mtp_head("Qwen3.5-9B-Q8_0.gguf"));
        assert!(!named_like_mtp_head("smtpd-model.gguf"));
        assert!(!named_like_mtp_head("mtp-notes.txt"));
    }

    #[test]
    fn pairs_a_model_with_a_fitting_head_in_its_folder() {
        let dir = std::env::temp_dir().join(format!("noullama-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("Winnow-E4B-Q8_0.gguf");
        let head = dir.join("winnow-mtp.gguf");
        std::fs::write(&model, gguf("gemma4", &[("block_count", 42), ("embedding_length", 2560)])).unwrap();
        std::fs::write(&head, gguf("gemma4-assistant", &[("block_count", 4), ("nextn_predict_layers", 4), ("embedding_length_out", 2560)])).unwrap();
        let found = model_mtp(&model.to_string_lossy());
        assert_eq!(found, ModelMtp { built_in: false, head: Some(head.to_string_lossy().into_owned()) });
        assert_eq!(model_mtp(&head.to_string_lossy()), ModelMtp::default());
        std::fs::remove_dir_all(dir).ok();
    }
}
