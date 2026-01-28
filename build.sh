#!/bin/bash

# Build WASM module for the poker tree builder webapp

set -e

echo "Building WASM module..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build the WASM package
cd crates/solver-wasm
wasm-pack build --target web --out-dir ../../webapp/pkg

echo ""
echo "Build complete!"
echo ""
echo "To run the webapp:"
echo "  cd webapp && python3 -m http.server 8080"
echo ""
echo "Then open http://localhost:8080 in your browser"
