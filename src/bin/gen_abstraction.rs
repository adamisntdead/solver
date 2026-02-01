//! CLI tool for generating hand abstractions.
//!
//! # Usage
//!
//! ```bash
//! # Generate river EHS abstraction with 500 buckets
//! cargo run --release --bin gen_abstraction -- \
//!   --street river \
//!   --type ehs \
//!   --buckets 500 \
//!   --output data/abstractions/river-EHS-500.abs
//!
//! # Generate turn EMD abstraction
//! cargo run --release --bin gen_abstraction -- \
//!   --street turn \
//!   --type emd \
//!   --buckets 5000 \
//!   --restarts 3 \
//!   --output data/abstractions/turn-EMD-5000.abs
//! ```

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use solver::poker::abstraction_gen::{AbstractionConfig, AbstractionType};
use solver::poker::abstraction_io::{default_filename, save_abstraction};
use solver::poker::indexer::Street;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_help();
        return;
    }

    // Parse arguments
    let mut street = Street::River;
    let mut abs_type = AbstractionType::EHS;
    let mut num_buckets = 200;
    let mut num_restarts = 5;
    let mut max_iterations = 100;
    let mut output_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--street" | "-s" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --street requires a value");
                    return;
                }
                street = parse_street(&args[i]).unwrap_or_else(|| {
                    eprintln!("Invalid street: {}", args[i]);
                    std::process::exit(1);
                });
            }
            "--type" | "-t" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --type requires a value");
                    return;
                }
                abs_type = parse_abstraction_type(&args[i]).unwrap_or_else(|| {
                    eprintln!("Invalid abstraction type: {}", args[i]);
                    std::process::exit(1);
                });
            }
            "--buckets" | "-b" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --buckets requires a value");
                    return;
                }
                num_buckets = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid bucket count: {}", args[i]);
                    std::process::exit(1);
                });
            }
            "--restarts" | "-r" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --restarts requires a value");
                    return;
                }
                num_restarts = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid restart count: {}", args[i]);
                    std::process::exit(1);
                });
            }
            "--iterations" | "-i" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --iterations requires a value");
                    return;
                }
                max_iterations = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid iteration count: {}", args[i]);
                    std::process::exit(1);
                });
            }
            "--output" | "-o" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --output requires a value");
                    return;
                }
                output_path = Some(PathBuf::from(&args[i]));
            }
            _ => {
                if args[i].starts_with('-') {
                    eprintln!("Unknown option: {}", args[i]);
                    return;
                }
            }
        }
        i += 1;
    }

    // Validate
    if !abs_type.is_valid_for_street(street) {
        eprintln!(
            "Error: {} is not valid for {}",
            abs_type.name(),
            street.name()
        );
        std::process::exit(1);
    }

    // Generate output path if not specified
    let output = output_path.unwrap_or_else(|| {
        PathBuf::from(default_filename(street, abs_type, num_buckets))
    });

    // Create parent directory if needed
    if let Some(parent) = output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!("Failed to create directory {}: {}", parent.display(), e);
                std::process::exit(1);
            });
        }
    }

    println!("Generating {} abstraction for {} street", abs_type.name(), street.name());
    println!("  Buckets: {}", num_buckets);
    println!("  Restarts: {}", num_restarts);
    println!("  Max iterations: {}", max_iterations);
    println!("  Output: {}", output.display());

    #[cfg(feature = "rayon")]
    println!("  Parallelism: {} threads (rayon)", rayon::current_num_threads());
    #[cfg(not(feature = "rayon"))]
    println!("  Parallelism: sequential (no rayon)");

    println!();

    // Configure
    let config = AbstractionConfig {
        street,
        abstraction_type: abs_type,
        num_buckets,
        num_restarts,
        max_iterations,
        ehs_squared: false,
        progress_callback: Some(Box::new(|pct| {
            eprint!("\rProgress: {}%  ", pct);
        })),
    };

    // Generate
    let start = Instant::now();

    #[cfg(feature = "rand")]
    let result = solver::poker::abstraction_gen::generate_street_abstraction(&config);

    #[cfg(not(feature = "rand"))]
    {
        eprintln!("Error: rand feature is required for abstraction generation");
        std::process::exit(1);
    }

    let elapsed = start.elapsed();
    eprintln!();

    #[cfg(feature = "rand")]
    {
        println!();
        println!("Generation complete:");
        println!("  Time: {:.2}s", elapsed.as_secs_f64());
        println!("  Boards processed: {}", result.num_boards);
        println!("  Total entries: {}", result.assignments.len());
        println!("  Actual buckets: {}", result.num_buckets);

        // Save
        println!();
        print!("Saving to {}...", output.display());
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match save_abstraction(&result, &output) {
            Ok(()) => {
                let file_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
                println!(" done ({} bytes)", file_size);
            }
            Err(e) => {
                eprintln!(" failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn print_help() {
    println!("gen_abstraction - Generate hand abstractions for poker solver");
    println!();
    println!("USAGE:");
    println!("    gen_abstraction [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -s, --street <STREET>      Street to generate for (preflop, flop, turn, river)");
    println!("    -t, --type <TYPE>          Abstraction type (ehs, ehs2, emd, asymemd, winsplit, aggsi, semiaggsi)");
    println!("    -b, --buckets <NUM>        Number of buckets (default: 200)");
    println!("    -r, --restarts <NUM>       K-means restarts (default: 5)");
    println!("    -i, --iterations <NUM>     Max iterations per restart (default: 100)");
    println!("    -o, --output <PATH>        Output file path");
    println!("    -h, --help                 Show this help message");
    println!();
    println!("ABSTRACTION TYPES:");
    println!("    ehs        Expected Hand Strength (all streets)");
    println!("    ehs2       EHS squared - captures variance (all streets)");
    println!("    emd        Earth Mover's Distance histogram (flop, turn)");
    println!("    asymemd    Asymmetric EMD - finer bins at high equity (flop, turn)");
    println!("    winsplit   Win/split frequency (river only)");
    println!("    aggsi      Aggressive suit isomorphism (all postflop)");
    println!("    semiaggsi  Semi-aggressive suit isomorphism (flop only)");
    println!();
    println!("EXAMPLES:");
    println!("    gen_abstraction -s river -t ehs -b 500 -o river-ehs-500.abs");
    println!("    gen_abstraction -s turn -t emd -b 5000 -r 3");
}

fn parse_street(s: &str) -> Option<Street> {
    match s.to_lowercase().as_str() {
        "preflop" | "pre" => Some(Street::Preflop),
        "flop" => Some(Street::Flop),
        "turn" => Some(Street::Turn),
        "river" => Some(Street::River),
        _ => None,
    }
}

fn parse_abstraction_type(s: &str) -> Option<AbstractionType> {
    match s.to_lowercase().as_str() {
        "ehs" => Some(AbstractionType::EHS),
        "ehs2" | "ehssquared" | "ehs-squared" => Some(AbstractionType::EHSSquared),
        "emd" => Some(AbstractionType::EMD),
        "asymemd" | "asym-emd" => Some(AbstractionType::AsymEMD),
        "winsplit" | "win-split" => Some(AbstractionType::WinSplit),
        "aggsi" | "agg-si" => Some(AbstractionType::AggSI),
        "semiaggsi" | "semi-agg-si" => Some(AbstractionType::SemiAggSI),
        _ => None,
    }
}
