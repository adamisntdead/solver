//! Convert Gambit abstraction files to our format.

use std::path::Path;

use solver::poker::abstraction_io::{load_gambit_abstraction, save_abstraction};

fn main() {
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
}

fn convert_single(input: &str, output: &str) {
    println!("Converting {} -> {}", input, output);

    let abs = load_gambit_abstraction(Path::new(input)).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}", input, e);
        std::process::exit(1);
    });

    println!(
        "  Loaded: {:?} {:?} with {} buckets, {} entries",
        abs.street,
        abs.abstraction_type,
        abs.num_buckets,
        abs.assignments.len()
    );

    save_abstraction(&abs, Path::new(output)).unwrap_or_else(|e| {
        eprintln!("Failed to save {}: {}", output, e);
        std::process::exit(1);
    });

    let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    println!("  Saved: {} bytes", file_size);
}

fn convert_all(gambit_dir: &str, output_dir: &str) {
    // Create output directory
    std::fs::create_dir_all(output_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create output directory: {}", e);
        std::process::exit(1);
    });

    // Known Gambit files
    let files = [
        ("flop-SEMI-AGG-SI.abs", "flop-SemiAggSI.abs"),
        ("turn-ASYMEMD2-AGGSI-64000.abs", "turn-AsymEMD-64000.abs"),
        (
            "river-WIN2SPLIT2-500-5RESTARTS.abs",
            "river-WinSplit-500.abs",
        ),
    ];

    for (input_name, output_name) in files {
        let input_path = format!("{}/{}", gambit_dir, input_name);
        let output_path = format!("{}/{}", output_dir, output_name);

        if Path::new(&input_path).exists() {
            convert_single(&input_path, &output_path);
        } else {
            println!("Skipping {} (not found)", input_path);
        }
    }

    println!("\nDone! Converted files are in {}", output_dir);
}
