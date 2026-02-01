//! Abstraction serialization and deserialization.
//!
//! File format for pre-built abstractions:
//!
//! ```text
//! Header (20 bytes):
//!   - magic: u32 = 0x41425354 ("ABST")
//!   - version: u16 = 1
//!   - street: u8 (0=preflop, 1=flop, 2=turn, 3=river)
//!   - abs_type: u8 (abstraction type enum)
//!   - num_buckets: u32
//!   - num_entries: u32
//!   - flags: u32 (bit 0 = has centers)
//!
//! Data (zstd compressed when zstd feature enabled):
//!   - assignments: [u16; num_entries]
//!   - centers (if has_centers flag):
//!     - For EHS: [f32; num_buckets]
//!     - For WinSplit: [[f32; 2]; num_buckets]
//!     - For EMD: [[f32; 50]; num_buckets]
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use solver::poker::abstraction_io::{save_abstraction, load_abstraction};
//!
//! // Save
//! save_abstraction(&generated, Path::new("river-ehs-500.abs"))?;
//!
//! // Load
//! let loaded = load_abstraction(Path::new("river-ehs-500.abs"))?;
//! ```

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::poker::abstraction_gen::{AbstractionType, CenterData, GeneratedAbstraction};
use crate::poker::ehs::EMD_NUM_BINS;
use crate::poker::indexer::Street;

/// Magic bytes for abstraction files.
const MAGIC: u32 = 0x41425354; // "ABST"

/// File format version.
const VERSION: u16 = 1;

/// Flag: file contains cluster centers.
const FLAG_HAS_CENTERS: u32 = 1;

/// Error type for abstraction IO.
#[derive(Debug)]
pub enum AbstractionIOError {
    /// IO error.
    Io(io::Error),
    /// Invalid magic bytes.
    InvalidMagic,
    /// Unsupported version.
    UnsupportedVersion(u16),
    /// Invalid street value.
    InvalidStreet(u8),
    /// Invalid abstraction type.
    InvalidAbstractionType(u8),
    /// Data corruption.
    DataCorruption(String),
}

impl From<io::Error> for AbstractionIOError {
    fn from(err: io::Error) -> Self {
        AbstractionIOError::Io(err)
    }
}

impl std::fmt::Display for AbstractionIOError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractionIOError::Io(e) => write!(f, "IO error: {}", e),
            AbstractionIOError::InvalidMagic => write!(f, "Invalid file magic"),
            AbstractionIOError::UnsupportedVersion(v) => {
                write!(f, "Unsupported version: {}", v)
            }
            AbstractionIOError::InvalidStreet(s) => write!(f, "Invalid street: {}", s),
            AbstractionIOError::InvalidAbstractionType(t) => {
                write!(f, "Invalid abstraction type: {}", t)
            }
            AbstractionIOError::DataCorruption(msg) => write!(f, "Data corruption: {}", msg),
        }
    }
}

impl std::error::Error for AbstractionIOError {}

/// Save a generated abstraction to file.
pub fn save_abstraction(
    abstraction: &GeneratedAbstraction,
    path: &Path,
) -> Result<(), AbstractionIOError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Determine flags
    let has_centers = abstraction.centers.is_some();
    let flags = if has_centers { FLAG_HAS_CENTERS } else { 0 };

    // Write header
    writer.write_all(&MAGIC.to_le_bytes())?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&[abstraction.street as u8])?;
    writer.write_all(&[abstraction_type_to_u8(abstraction.abstraction_type)])?;
    writer.write_all(&(abstraction.num_buckets as u32).to_le_bytes())?;
    writer.write_all(&(abstraction.assignments.len() as u32).to_le_bytes())?;
    writer.write_all(&flags.to_le_bytes())?;

    // Prepare data
    let mut data = Vec::new();

    // Write assignments
    for &assignment in &abstraction.assignments {
        data.extend_from_slice(&assignment.to_le_bytes());
    }

    // Write centers if present
    if let Some(ref centers) = abstraction.centers {
        write_centers(&mut data, centers)?;
    }

    // Compress and write
    #[cfg(feature = "zstd")]
    {
        let compressed = zstd::encode_all(&data[..], 3)?;
        writer.write_all(&(compressed.len() as u32).to_le_bytes())?;
        writer.write_all(&compressed)?;
    }

    #[cfg(not(feature = "zstd"))]
    {
        writer.write_all(&(data.len() as u32).to_le_bytes())?;
        writer.write_all(&data)?;
    }

    writer.flush()?;
    Ok(())
}

/// Load a generated abstraction from file.
pub fn load_abstraction(path: &Path) -> Result<GeneratedAbstraction, AbstractionIOError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut header = [0u8; 20];
    reader.read_exact(&mut header)?;

    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != MAGIC {
        return Err(AbstractionIOError::InvalidMagic);
    }

    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != VERSION {
        return Err(AbstractionIOError::UnsupportedVersion(version));
    }

    let street = u8_to_street(header[6])?;
    let abstraction_type = u8_to_abstraction_type(header[7])?;
    let num_buckets = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let num_entries = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
    let flags = u32::from_le_bytes([header[16], header[17], header[18], header[19]]);

    let has_centers = (flags & FLAG_HAS_CENTERS) != 0;

    // Read compressed data size
    let mut size_bytes = [0u8; 4];
    reader.read_exact(&mut size_bytes)?;
    let data_size = u32::from_le_bytes(size_bytes) as usize;

    // Read compressed data
    let mut compressed = vec![0u8; data_size];
    reader.read_exact(&mut compressed)?;

    // Decompress
    #[cfg(feature = "zstd")]
    let data = zstd::decode_all(&compressed[..])?;

    #[cfg(not(feature = "zstd"))]
    let data = compressed;

    // Parse assignments
    let assignments_bytes = num_entries * 2;
    if data.len() < assignments_bytes {
        return Err(AbstractionIOError::DataCorruption(
            "Not enough data for assignments".to_string(),
        ));
    }

    let mut assignments = Vec::with_capacity(num_entries);
    for i in 0..num_entries {
        let offset = i * 2;
        let value = u16::from_le_bytes([data[offset], data[offset + 1]]);
        assignments.push(value);
    }

    // Parse centers if present
    let centers = if has_centers {
        let center_data = &data[assignments_bytes..];
        Some(read_centers(center_data, abstraction_type, num_buckets)?)
    } else {
        None
    };

    Ok(GeneratedAbstraction {
        street,
        abstraction_type,
        num_buckets,
        assignments,
        centers,
        num_boards: 0, // Not stored in file
    })
}

/// Write center data to buffer.
fn write_centers(data: &mut Vec<u8>, centers: &CenterData) -> Result<(), AbstractionIOError> {
    match centers {
        CenterData::Scalar(values) => {
            for &v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        CenterData::TwoD(values) => {
            for v in values {
                for &x in v {
                    data.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        CenterData::Histogram(values) => {
            for v in values {
                for &x in v {
                    data.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
    Ok(())
}

/// Read center data from buffer.
fn read_centers(
    data: &[u8],
    abstraction_type: AbstractionType,
    num_buckets: usize,
) -> Result<CenterData, AbstractionIOError> {
    match abstraction_type {
        AbstractionType::EHS | AbstractionType::EHSSquared => {
            let expected = num_buckets * 4;
            if data.len() < expected {
                return Err(AbstractionIOError::DataCorruption(
                    "Not enough data for EHS centers".to_string(),
                ));
            }

            let mut values = Vec::with_capacity(num_buckets);
            for i in 0..num_buckets {
                let offset = i * 4;
                let v = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                values.push(v);
            }
            Ok(CenterData::Scalar(values))
        }
        AbstractionType::WinSplit => {
            let expected = num_buckets * 8; // 2 * f32
            if data.len() < expected {
                return Err(AbstractionIOError::DataCorruption(
                    "Not enough data for WinSplit centers".to_string(),
                ));
            }

            let mut values = Vec::with_capacity(num_buckets);
            for i in 0..num_buckets {
                let offset = i * 8;
                let x = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                let y = f32::from_le_bytes([
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]);
                values.push([x, y]);
            }
            Ok(CenterData::TwoD(values))
        }
        AbstractionType::EMD | AbstractionType::AsymEMD => {
            let expected = num_buckets * EMD_NUM_BINS * 4;
            if data.len() < expected {
                return Err(AbstractionIOError::DataCorruption(
                    "Not enough data for EMD centers".to_string(),
                ));
            }

            let mut values = Vec::with_capacity(num_buckets);
            for i in 0..num_buckets {
                let mut hist = [0.0f32; EMD_NUM_BINS];
                for j in 0..EMD_NUM_BINS {
                    let offset = (i * EMD_NUM_BINS + j) * 4;
                    hist[j] = f32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                }
                values.push(hist);
            }
            Ok(CenterData::Histogram(values))
        }
        _ => Ok(CenterData::Scalar(vec![])), // No centers for deterministic types
    }
}

/// Convert abstraction type to u8.
fn abstraction_type_to_u8(t: AbstractionType) -> u8 {
    match t {
        AbstractionType::EHS => 0,
        AbstractionType::EHSSquared => 1,
        AbstractionType::EMD => 2,
        AbstractionType::AsymEMD => 3,
        AbstractionType::WinSplit => 4,
        AbstractionType::AggSI => 5,
        AbstractionType::SemiAggSI => 6,
    }
}

/// Convert u8 to abstraction type.
fn u8_to_abstraction_type(v: u8) -> Result<AbstractionType, AbstractionIOError> {
    match v {
        0 => Ok(AbstractionType::EHS),
        1 => Ok(AbstractionType::EHSSquared),
        2 => Ok(AbstractionType::EMD),
        3 => Ok(AbstractionType::AsymEMD),
        4 => Ok(AbstractionType::WinSplit),
        5 => Ok(AbstractionType::AggSI),
        6 => Ok(AbstractionType::SemiAggSI),
        _ => Err(AbstractionIOError::InvalidAbstractionType(v)),
    }
}

/// Convert u8 to street.
fn u8_to_street(v: u8) -> Result<Street, AbstractionIOError> {
    match v {
        0 => Ok(Street::Preflop),
        1 => Ok(Street::Flop),
        2 => Ok(Street::Turn),
        3 => Ok(Street::River),
        _ => Err(AbstractionIOError::InvalidStreet(v)),
    }
}

/// Get file extension for abstraction files.
pub fn abstraction_file_extension() -> &'static str {
    "abs"
}

/// Generate a default filename for an abstraction.
pub fn default_filename(
    street: Street,
    abstraction_type: AbstractionType,
    num_buckets: usize,
) -> String {
    format!(
        "{}-{}-{}.{}",
        street.name(),
        abstraction_type.name(),
        num_buckets,
        abstraction_file_extension()
    )
}

/// Load an abstraction file in Gambit format.
///
/// Gambit's .abs format:
/// - round_id: i32 (0=preflop, 1=flop, 2=turn, 3=river)
/// - num_buckets: i32
/// - compressed_size: usize (8 bytes)
/// - data: zstd-compressed uint32_t bucket assignments
///
/// Note: Gambit uses uint32_t for bucket assignments while we use u16.
/// This function will fail if any bucket value exceeds u16::MAX.
#[cfg(feature = "zstd")]
pub fn load_gambit_abstraction(path: &Path) -> Result<GeneratedAbstraction, AbstractionIOError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read round_id (i32)
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let round_id = i32::from_le_bytes(buf4);
    let street = match round_id {
        0 => Street::Preflop,
        1 => Street::Flop,
        2 => Street::Turn,
        3 => Street::River,
        _ => return Err(AbstractionIOError::InvalidStreet(round_id as u8)),
    };

    // Read num_buckets (i32)
    reader.read_exact(&mut buf4)?;
    let num_buckets = i32::from_le_bytes(buf4) as usize;

    // Read compressed_size (usize = 8 bytes on 64-bit)
    let mut buf8 = [0u8; 8];
    reader.read_exact(&mut buf8)?;
    let compressed_size = usize::from_le_bytes(buf8);

    // Read compressed data
    let mut compressed = vec![0u8; compressed_size];
    reader.read_exact(&mut compressed)?;

    // Decompress using zstd
    let decompressed = zstd::decode_all(&compressed[..])?;

    // Parse as u32 values (Gambit uses uint32_t)
    if decompressed.len() % 4 != 0 {
        return Err(AbstractionIOError::DataCorruption(
            "Decompressed data size not multiple of 4".to_string(),
        ));
    }

    let num_entries = decompressed.len() / 4;
    let mut assignments = Vec::with_capacity(num_entries);

    for i in 0..num_entries {
        let offset = i * 4;
        let value = u32::from_le_bytes([
            decompressed[offset],
            decompressed[offset + 1],
            decompressed[offset + 2],
            decompressed[offset + 3],
        ]);

        // Convert u32 to u16, failing if value exceeds u16::MAX
        if value > u16::MAX as u32 {
            return Err(AbstractionIOError::DataCorruption(format!(
                "Bucket value {} exceeds u16::MAX at index {}",
                value, i
            )));
        }
        assignments.push(value as u16);
    }

    // Infer abstraction type from filename or use a default
    // Gambit doesn't store the type in the file, so we use a heuristic
    let abstraction_type = infer_gambit_abstraction_type(path, street, num_buckets);

    Ok(GeneratedAbstraction {
        street,
        abstraction_type,
        num_buckets,
        assignments,
        centers: None, // Gambit doesn't store centers in file
        num_boards: 0, // Not stored in Gambit format
    })
}

/// Infer abstraction type from Gambit filename.
#[cfg(feature = "zstd")]
fn infer_gambit_abstraction_type(
    path: &Path,
    street: Street,
    _num_buckets: usize,
) -> AbstractionType {
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    if filename.contains("semi-agg-si") || filename.contains("semiaggsi") {
        AbstractionType::SemiAggSI
    } else if filename.contains("asymemd") || filename.contains("asym-emd") {
        // Check asymemd before aggsi since filenames like "ASYMEMD-AGGSI" should be AsymEMD
        AbstractionType::AsymEMD
    } else if filename.contains("agg-si") || filename.contains("aggsi") {
        AbstractionType::AggSI
    } else if filename.contains("emd") {
        AbstractionType::EMD
    } else if filename.contains("win") || filename.contains("split") {
        AbstractionType::WinSplit
    } else if filename.contains("ehs2") || filename.contains("ehssquared") {
        AbstractionType::EHSSquared
    } else if filename.contains("ehs") {
        AbstractionType::EHS
    } else {
        // Default based on street
        match street {
            Street::Flop => AbstractionType::SemiAggSI,
            Street::Turn => AbstractionType::AsymEMD,
            Street::River => AbstractionType::WinSplit,
            Street::Preflop => AbstractionType::EHS,
        }
    }
}

/// Detect if a file is in Gambit format by checking header.
pub fn is_gambit_format(path: &Path) -> Result<bool, AbstractionIOError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read first 4 bytes
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;

    let magic = u32::from_le_bytes(buf);

    // Our format starts with MAGIC (0x41425354 = "ABST")
    // Gambit format starts with round_id (0, 1, 2, or 3)
    Ok(magic != MAGIC && magic <= 3)
}

/// Load an abstraction file, auto-detecting format (ours or Gambit's).
#[cfg(feature = "zstd")]
pub fn load_abstraction_auto(path: &Path) -> Result<GeneratedAbstraction, AbstractionIOError> {
    if is_gambit_format(path)? {
        load_gambit_abstraction(path)
    } else {
        load_abstraction(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_conversion_roundtrip() {
        for t in [
            AbstractionType::EHS,
            AbstractionType::EHSSquared,
            AbstractionType::EMD,
            AbstractionType::AsymEMD,
            AbstractionType::WinSplit,
            AbstractionType::AggSI,
            AbstractionType::SemiAggSI,
        ] {
            let v = abstraction_type_to_u8(t);
            let t2 = u8_to_abstraction_type(v).unwrap();
            assert_eq!(t, t2);
        }
    }

    #[test]
    fn test_street_conversion_roundtrip() {
        for s in [
            Street::Preflop,
            Street::Flop,
            Street::Turn,
            Street::River,
        ] {
            let v = s as u8;
            let s2 = u8_to_street(v).unwrap();
            assert_eq!(s, s2);
        }
    }

    #[test]
    fn test_save_load_roundtrip() {
        // Use a unique temp file path
        let path = std::env::temp_dir().join(format!("test_abstraction_{}.abs", std::process::id()));

        let original = GeneratedAbstraction {
            street: Street::River,
            abstraction_type: AbstractionType::EHS,
            num_buckets: 10,
            assignments: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            centers: Some(CenterData::Scalar(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
            num_boards: 100,
        };

        save_abstraction(&original, &path).unwrap();

        let loaded = load_abstraction(&path).unwrap();

        // Clean up
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.street, original.street);
        assert_eq!(loaded.abstraction_type, original.abstraction_type);
        assert_eq!(loaded.num_buckets, original.num_buckets);
        assert_eq!(loaded.assignments, original.assignments);

        if let (Some(CenterData::Scalar(orig)), Some(CenterData::Scalar(load))) =
            (&original.centers, &loaded.centers)
        {
            assert_eq!(orig.len(), load.len());
            for (a, b) in orig.iter().zip(load.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_default_filename() {
        let name = default_filename(Street::River, AbstractionType::EHS, 500);
        assert_eq!(name, "river-EHS-500.abs");
    }

    #[test]
    fn test_infer_gambit_abstraction_type() {
        use std::path::PathBuf;

        let path = PathBuf::from("flop-SEMI-AGG-SI.abs");
        assert_eq!(
            infer_gambit_abstraction_type(&path, Street::Flop, 1170),
            AbstractionType::SemiAggSI
        );

        // AsymEMD should be detected even with AGGSI in the filename
        let path = PathBuf::from("turn-ASYMEMD2-AGGSI-64000.abs");
        assert_eq!(
            infer_gambit_abstraction_type(&path, Street::Turn, 64000),
            AbstractionType::AsymEMD
        );

        // Pure AggSI file
        let path = PathBuf::from("turn-AGGSI-1000.abs");
        assert_eq!(
            infer_gambit_abstraction_type(&path, Street::Turn, 1000),
            AbstractionType::AggSI
        );

        let path = PathBuf::from("river-WIN2SPLIT2-500.abs");
        assert_eq!(
            infer_gambit_abstraction_type(&path, Street::River, 500),
            AbstractionType::WinSplit
        );
    }

    #[test]
    fn test_is_gambit_format_detection() {
        // Create temp file with our format
        let our_path =
            std::env::temp_dir().join(format!("test_our_format_{}.abs", std::process::id()));
        {
            let mut file = File::create(&our_path).unwrap();
            file.write_all(&MAGIC.to_le_bytes()).unwrap();
            file.write_all(&[0u8; 16]).unwrap(); // padding
        }
        assert!(!is_gambit_format(&our_path).unwrap());
        let _ = std::fs::remove_file(&our_path);

        // Create temp file with Gambit format (round_id = 1 for flop)
        let gambit_path =
            std::env::temp_dir().join(format!("test_gambit_format_{}.abs", std::process::id()));
        {
            let mut file = File::create(&gambit_path).unwrap();
            file.write_all(&1i32.to_le_bytes()).unwrap(); // round_id = FLOP
            file.write_all(&[0u8; 16]).unwrap(); // padding
        }
        assert!(is_gambit_format(&gambit_path).unwrap());
        let _ = std::fs::remove_file(&gambit_path);
    }
}
