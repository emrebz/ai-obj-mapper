# Basic Usage Examples

## Quick Start

### 1. Basic Rendering
```bash
python ai_obj_mapper.py models/example_building.obj output/basic/
```

This will:
- Generate an AI facade texture (512x512px)
- Render 3 perspective views
- Save results in `output/basic/`

### 2. High-Resolution Textures
```bash
python ai_obj_mapper.py models/example_building.obj output/hires/ --resolution 1024
```

### 3. Force GPU/CPU
```bash
# Force CUDA GPU
python ai_obj_mapper.py models/example_building.obj output/gpu/ --device cuda

# Force CPU (slower but works without GPU)
python ai_obj_mapper.py models/example_building.obj output/cpu/ --device cpu

# Force Apple Silicon GPU
python ai_obj_mapper.py models/example_building.obj output/mps/ --device mps
```

## Expected Output

After running, you'll find:
- `perspective_front_right.png` - 30° view
- `perspective_front_left.png` - 150° view  
- `perspective_side_profile.png` - 270° view
- `ai_detailed_model.blend` - Blender scene file
- `ai_textures/ai_facade_texture.png` - Generated texture

## Performance Tips

- **GPU Memory**: 8GB+ recommended for 1024px textures
- **Processing Time**: 2-5 minutes for 512px, 5-10 minutes for 1024px
- **Model Size**: Works best with 1K-10K face models 