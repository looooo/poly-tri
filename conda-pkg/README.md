# Conda Recipe for PolyTri

This directory contains a Conda recipe for building and packaging the PolyTri library.

## Structure

- `meta.yaml` - Main recipe file with package metadata and dependencies
- `build.sh` - Build script for Unix-like systems (Linux, macOS)
- `bld.bat` - Build script for Windows
- `README.md` - This file

## Building the Package

### Prerequisites

- Conda or Mamba installed
- Conda-build installed: `conda install conda-build`

### Build Commands

From the project root directory:

```bash
# Build the package
conda build conda-pkg

# Build for specific platforms
conda build conda-pkg --platform linux-64
conda build conda-pkg --platform osx-64
conda build conda-pkg --platform osx-arm64
conda build conda-pkg --platform win-64

# Build and test
conda build conda-pkg --test

# Install locally after build
conda install --use-local polytri
```

### Using Mamba (faster)

```bash
mamba build conda-pkg
```

## Package Information

- **Name**: polytri
- **Version**: 0.1.0
- **Dependencies**:
  - Python >=3.8
  - numpy
  - Rust (build only)
  - maturin (build only)

## Build Process

1. **Rust Build**: The Rust extension is built using `maturin` with the `python` feature enabled
2. **Wheel Installation**: The built wheel is installed via pip
3. **Python Files**: Python package files are copied to site-packages

## Testing

The recipe includes automatic tests that verify:
- All modules can be imported (`polytri`, `polytri._python`, `polytri._rust`)
- The main `PolyTri` class can be imported
- The Python fallback `PolyTriPy` can be imported
- At least one implementation (Rust or Python) is available

## Uploading to Conda-Forge

To submit this package to conda-forge:

1. Fork the [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes) repository
2. Copy the recipe files to `recipes/polytri/`
3. Submit a pull request

## Local Development

For local development and testing:

```bash
# Create a test environment
conda create -n polytri-test python=3.11 numpy
conda activate polytri-test

# Build and install
conda build conda-pkg
conda install --use-local polytri

# Test
python -c "from polytri import PolyTri; print('Success!')"
```

## Notes

- The build process requires Rust and maturin, which are specified as build dependencies
- The Rust extension is built in release mode for optimal performance
- Python package files are copied separately to ensure the full package structure is available
- The recipe supports all major platforms: Linux, macOS (Intel and ARM), and Windows

