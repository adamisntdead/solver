//! Convert Gambit abstraction files to our format.
//!
//! Requires the `zstd` feature to be enabled.

#[cfg(not(feature = "zstd"))]
fn main() {
    eprintln!("Error: This binary requires the 'zstd' feature.");
    eprintln!("Run with: cargo run --bin convert_gambit --features zstd -- <args>");
    std::process::exit(1);
}

#[cfg(feature = "zstd")]
fn main() {
    use std::path::Path;
    use solver::poker::abstraction_io::{load_gambit_abstraction, save_abstraction};

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: convert_gambit <input.abs> <output.abs>");
        eprintln!("       convert_gambit --all <gambit_dir> <output_dir>");
        std::process::exit(1);
    }

    if args[1] == "--all" {
        if args.len() < 4 {
            eprintln!("Usage: convert_gambit --all <gambit_dir> <output_dir>");
            std::process::exit(1);
        }
        convert_all(&args[2], &args[3]);
    } else {
        convert_single(&args[1], &args[2]);
    }

    fn convert_single(input: &str, output: &str) {
        println!("Converting {} -> {}", input, output);

        let abs = load_gambit_abstraction(Path::new(input)).unwrap_or_else(|e| {
            eprintln!("Failed to load {}: {:?}", input, e);
            std::process::exit(1);
        });

        println!(
            "  Loaded: street={:?}, type={:?}, buckets={}, entries={}",
            abs.street, abs.abs_type, abs.num_buckets, abs.assignments.len()
        );

        save_abstraction(Path::new(output), &abs).unwrap_or_else(|e| {
            eprintln!("Failed to save {}: {:?}", output, e);
            std::process::exit(1);
        });

        // Show file size
        let size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        println!("  Saved: {} bytes ({:.1} KB)", size, size as f64 / 1024.0);
    }

    fn convert_all(gambit_dir: &str, output_dir: &str) {
        let files = [
            "flop-SEMI-AGG-SI.abs",
            "turn-ASYMEMD2-AGGSI-64000.abs",
            "river-WIN2SPLIT2-500.abs",
        ];

        // Create output directory if needed
        std::fs::create_dir_all(output_dir).unwrap_or_else(|e| {
            eprintln!("Failed to create {}: {}", output_dir, e);
            std::process::exit(1);
        });

        // Map to our output filenames
        let output_names = [
            "flop-SemiAggSI.abs",
            "turn-AsymEMD-64000.abs",
            "river-WinSplit-500.abs",
        ];

        for (input_name, output_name) in files.iter().zip(output_names.iter()) {
            let input_path = format!("{}/{}", gambit_dir, input_name);
            let output_path = format!("{}/{}", output_dir, output_name);
            convert_single(&input_path, &output_path);
        }

        println!("\nAll files converted successfully!");
    }
}
