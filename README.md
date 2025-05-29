# AI OBJ Mapper

🎨 **AI-Powered 3D Model Texture Generator and Renderer**

A standalone Python tool that automatically generates realistic AI textures for 3D OBJ models and renders them from multiple perspectives using Blender.

## ✨ Features

- **AI Texture Generation**: Creates photorealistic building facade textures using Stable Diffusion
- **Multi-Perspective Rendering**: Automatically renders models from 3 different camera angles
- **Seamless Texture Processing**: Makes textures tileable for better application
- **Professional Lighting**: Urban sky environment with realistic lighting setup
- **High-Quality Output**: Cycles rendering engine with denoising for crisp results
- **Multiple AI Models**: Supports RealVisXL, DreamShaper XL, and SDXL fallbacks

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Blender 3.0+ (installed and available in PATH)
- CUDA-compatible GPU (recommended) or Apple Silicon Mac

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ai-obj-mapper
```

2. Install dependencies:
```bash
pip install -r standalone_requirements.txt
```

3. Run the example:
```bash
python ai_obj_mapper.py models/example_building.obj output/
```

## 📊 Usage

### Basic Usage

```bash
python ai_obj_mapper.py <input.obj> <output_directory> [options]
```

### Options

- `--resolution`: AI texture resolution (default: 512px)
- `--device`: Force AI device (cuda, mps, cpu)

### Example Commands

```bash
# Basic usage with default settings
python ai_obj_mapper.py models/building.obj renders/

# High-resolution texture generation
python ai_obj_mapper.py models/building.obj renders/ --resolution 1024

# Force specific device
python ai_obj_mapper.py models/building.obj renders/ --device cuda
```

## 📁 Output Files

The tool generates:

- **3 Perspective Renders**: Front-right (30°), Front-left (150°), Side profile (270°)
- **AI-Generated Texture**: High-quality facade texture
- **Blender Scene**: Complete .blend file for further editing

## 🎯 Supported Model Types

- **OBJ Files**: Standard Wavefront OBJ format
- **Building Models**: Optimized for architectural structures
- **Complex Geometry**: Handles models with thousands of faces

## 🛠️ Technical Details

### AI Models Used
1. **RealVisXL V4.0** (Primary): Architecture-focused realistic generation
2. **DreamShaper XL** (Fallback): High-quality general purpose model
3. **SDXL Base** (Final fallback): Stable Diffusion XL base model

### Rendering Pipeline
- **Engine**: Blender Cycles with GPU acceleration
- **Samples**: 64 samples with denoising
- **Resolution**: 1920x1080 output images
- **Environment**: Nishita sky model for realistic lighting

## 📖 Examples

Check the `examples/` directory for:
- Sample OBJ files
- Expected output renders
- Usage tutorials

## 🔧 Troubleshooting

### Common Issues

**AI Model Loading Fails**
- Ensure you have sufficient GPU memory (8GB+ recommended)
- Try reducing texture resolution with `--resolution 512`

**Blender Not Found**
- Make sure Blender is installed and available in your system PATH
- On macOS: Add `/Applications/Blender.app/Contents/MacOS/` to PATH

**Out of Memory**
- Reduce texture resolution
- Use CPU mode with `--device cpu`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion models
- Blender Foundation for the amazing 3D software
- Hugging Face for the diffusers library

---

**Made with ❤️ for the 3D visualization community** 