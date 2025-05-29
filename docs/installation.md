# Installation Guide

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8GB RAM
- 2GB free disk space
- Blender 3.0+ installed

### Recommended Requirements
- Python 3.9+
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (for CUDA)
- Apple Silicon Mac (for MPS acceleration)
- 10GB free disk space

## Step-by-Step Installation

### 1. Install Python Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd ai-obj-mapper

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r standalone_requirements.txt
```

### 2. Install Blender

#### macOS
```bash
# Using Homebrew
brew install --cask blender

# Or download from https://www.blender.org/download/
# Add to PATH: export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install blender
```

#### Windows
1. Download from https://www.blender.org/download/
2. Install and add to system PATH

### 3. GPU Setup (Optional but Recommended)

#### NVIDIA CUDA
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Apple Silicon (MPS)
```bash
# MPS support is included in default PyTorch installation
# No additional setup needed
```

### 4. Verify Installation

```bash
# Test basic functionality
python ai_obj_mapper.py --help

# Test with example model
python ai_obj_mapper.py models/example_building.obj output/test/
```

## Troubleshooting

### Common Issues

**"blender: command not found"**
- Ensure Blender is installed and in your system PATH
- On macOS: `export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"`

**"CUDA out of memory"**
- Reduce texture resolution: `--resolution 512`
- Use CPU mode: `--device cpu`

**"No module named 'diffusers'"**
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r standalone_requirements.txt`

**Slow AI generation**
- Ensure GPU acceleration is working
- Check device with: `python -c "import torch; print(torch.cuda.is_available())"` 