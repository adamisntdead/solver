#!/bin/bash
# Generate standard hand abstractions for poker solver
#
# Usage:
#   ./scripts/generate_abstractions.sh           # Generate all
#   ./scripts/generate_abstractions.sh flop      # Generate flop only
#   ./scripts/generate_abstractions.sh turn      # Generate turn only
#   ./scripts/generate_abstractions.sh river     # Generate river only
#
# Configuration matches Gambit's standard abstractions:
#   Flop:  SemiAggSI (1170 buckets, deterministic)
#   Turn:  AsymEMD (64000 buckets)
#   River: WinSplit (500 buckets)
#
# Note: Turn generation is expensive (~30-60 minutes on modern hardware)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/data/abstractions"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Parse arguments
TARGET="${1:-all}"

generate_flop() {
    echo "=== Generating Flop Abstraction ==="
    echo "Type: SemiAggSI (deterministic, no clustering)"
    echo "Expected buckets: ~1170"
    echo ""

    cargo run --release --bin gen_abstraction -- \
        --street flop \
        --type semiaggsi \
        --output "$OUTPUT_DIR/flop-SEMI-AGG-SI.abs"

    echo ""
    echo "Flop abstraction complete."
    echo ""
}

generate_turn() {
    echo "=== Generating Turn Abstraction ==="
    echo "Type: AsymEMD with 64000 buckets"
    echo "Warning: This may take 30-60 minutes"
    echo ""

    cargo run --release --bin gen_abstraction -- \
        --street turn \
        --type asymemd \
        --buckets 64000 \
        --restarts 5 \
        --iterations 100 \
        --output "$OUTPUT_DIR/turn-ASYMEMD-64000.abs"

    echo ""
    echo "Turn abstraction complete."
    echo ""
}

generate_river() {
    echo "=== Generating River Abstraction ==="
    echo "Type: WinSplit with 500 buckets"
    echo ""

    cargo run --release --bin gen_abstraction -- \
        --street river \
        --type winsplit \
        --buckets 500 \
        --restarts 5 \
        --iterations 100 \
        --output "$OUTPUT_DIR/river-WINSPLIT-500.abs"

    echo ""
    echo "River abstraction complete."
    echo ""
}

# Main execution
case "$TARGET" in
    flop)
        generate_flop
        ;;
    turn)
        generate_turn
        ;;
    river)
        generate_river
        ;;
    all)
        echo "Generating all standard abstractions..."
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        generate_flop
        generate_river
        generate_turn  # Turn last since it's slowest
        echo "=== All abstractions generated ==="
        ls -la "$OUTPUT_DIR"
        ;;
    *)
        echo "Usage: $0 [flop|turn|river|all]"
        exit 1
        ;;
esac
