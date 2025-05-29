#!/usr/bin/env python3
"""
AI-Detailed Multi-Perspective OBJ Mapper
Generates detailed facade texture with AI and applies it in Blender.
Renders from 3 different camera angles with an urban sky.
"""

import subprocess
import tempfile
from pathlib import Path
import argparse
import os
import time
import math
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np
from typing import Optional

class AiDetailedObjMapper:
    """Generates AI facade texture and renders OBJ from multiple perspectives."""
    
    def __init__(self, obj_path: str, output_dir: str, device: str = None):
        """Initialize the mapper."""
        self.obj_path = Path(obj_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.texture_dir = self.output_dir / "ai_textures"
        self.texture_dir.mkdir(exist_ok=True, parents=True)
        
        # Choose appropriate device for AI
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.pipe = None
        self.ai_texture_path = None

    def load_ai_pipeline(self):
        """Load AI pipeline for texture generation."""
        if self.pipe:
            return
        print(f"🔄 Loading AI model for texture generation on {self.device}...")
        
        # Try RealVisXL first (best for realistic facades)
        try:
            model_id = "SG161222/RealVisXL_V4.0"
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            ).to(self.device)
            print(f"✅ RealVisXL (Architecture-focused) loaded on {self.device}")
            
        except Exception as e:
            print(f"⚠️ RealVisXL failed: {e}, trying DreamShaper...")
            try:
                model_id = "Lykon/dreamshaper-xl-1-0"
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device == "cuda" else None
                ).to(self.device)
                print(f"✅ DreamShaper XL loaded on {self.device}")
                
            except Exception as e2:
                print(f"⚠️ DreamShaper failed: {e2}, trying SDXL...")
                try:
                    self.pipe = DiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        use_safetensors=True
                    ).to(self.device)
                    print(f"✅ SDXL Base loaded on {self.device}")
                except Exception as e_sdxl:
                    print(f"❌ All models failed. AI Texture generation will not be available.")
                    self.pipe = None

    def make_seamless(self, image: Image.Image) -> Image.Image:
        """Make texture seamless by blending edges."""
        width, height = image.size
        img_np = np.array(image).astype(np.float32)
        
        # Create a mask for blending edges - more gradual blend
        mask = np.ones((height, width, 3), dtype=np.float32)
        blend_width_h = int(width * 0.15) # Blend 15% from horizontal edges
        blend_width_v = int(height * 0.15) # Blend 15% from vertical edges

        for i in range(blend_width_h):
            factor = i / blend_width_h
            mask[:, i, :] *= factor
            mask[:, width - 1 - i, :] *= factor
        
        for i in range(blend_width_v):
            factor = i / blend_width_v
            mask[i, :, :] *= factor
            mask[height - 1 - i, :, :] *= factor
        
        # Create flipped versions for blending
        top_half = img_np[:height//2, :, :]
        bottom_half = img_np[height//2:, :, :]
        left_half = img_np[:, :width//2, :]
        right_half = img_np[:, width//2:, :]

        # Horizontal blend
        h_blend_area = np.concatenate((right_half, left_half), axis=1)
        img_np = img_np * mask + h_blend_area * (1 - mask)
        
        # Vertical blend (using the already horizontally blended image)
        v_blend_area = np.concatenate((bottom_half, top_half), axis=0)
        img_np = img_np * mask + v_blend_area * (1-mask)
        
        seamless_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        return seamless_img

    def generate_facade_texture(self, resolution: int = 1024) -> Path:
        """Generate a detailed building facade texture using AI."""
        if self.pipe is None:
            self.load_ai_pipeline()
            if self.pipe is None: # Still None after trying to load
                 print("❌ AI Pipeline not available. Cannot generate texture.")
                 return None

        prompt = (
            "crisp photorealistic urban building facade, clean modern architecture, "
            "detailed realistic windows with glass reflections, concrete and brick materials, "
            "sharp architectural details, professional architectural photography, "
            "high resolution texture, clean lines, realistic lighting, architectural realism"
        )
        negative_prompt = (
            "cartoon, anime, painting, sketch, unrealistic, blurry, low quality, watermark, text, "
            "oversaturated, deformed, ugly windows, abstract, artistic, spikes, artifacts, "
            "repetitive patterns, graffiti, people, cars, vegetation, distorted geometry"
        )
        
        print(f"🎨 Generating AI facade texture ({resolution}x{resolution}px)...")
        
        try:
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=resolution,
                    height=resolution,
                    num_inference_steps=30, # Slightly more steps for architectural detail
                    guidance_scale=6.0, # Lower guidance for more realistic results
                ).images[0]
            
            print("✨ AI image generated, making it seamless...")
            # Make texture seamless (optional, but good for tiling)
            # result = self.make_seamless(result)

            self.ai_texture_path = self.texture_dir / "ai_facade_texture.png"
            result.save(self.ai_texture_path, quality=95)
            print(f"✅ AI facade texture saved to: {self.ai_texture_path}")
            return self.ai_texture_path
        except Exception as e:
            print(f"❌ Error during AI texture generation: {e}")
            return None

    def parse_obj(self):
        """Parse OBJ file to extract vertices and faces."""
        vertices = []
        faces = []
        
        print(f"📁 Reading OBJ file: {self.obj_path}")
        
        with open(self.obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()[1:4]
                    vertex = [float(p) for p in parts]
                    vertices.append(vertex)
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    face_indices = []
                    for part in parts:
                        vertex_idx = int(part.split('//')[0].split('/')[0]) - 1
                        face_indices.append(vertex_idx)
                    faces.append(face_indices)
        
        print(f"✅ Parsed: {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces
    
    def create_blender_script(self, vertices, faces, ai_texture_path_str: Optional[str]) -> str:
        """Create Blender script to build geometry, apply AI texture, and render 3 perspectives."""
        
        output_dir_str = str(self.output_dir.resolve())
        vertices_str = str(vertices)
        faces_str = str(faces)
        
        script = f'''
import bpy
import bmesh
from mathutils import Vector
import math
import sys
import os

print("🔄 Starting AI-detailed multi-perspective OBJ mapping...")

try:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    print("✅ Cleared scene")

    vertices = {vertices_str}
    faces = {faces_str}
    
    print(f"Building mesh with {{len(vertices)}} vertices and {{len(faces)}} faces")
    
    mesh = bpy.data.meshes.new("OBJ_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    
    obj = bpy.data.objects.new("OBJ_Building", mesh)
    bpy.context.collection.objects.link(obj)
    
    obj.rotation_euler = (math.radians(-270), 0, 0)
    bpy.context.view_layer.update()
    
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z_rotated = min(corner.z for corner in bbox_corners)
    obj.location = (0, 0, -min_z_rotated)
    bpy.context.view_layer.update()
    
    print("✅ Mesh created, rotated, and grounded")

    bpy.context.scene.render.engine = 'CYCLES'
    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.image_settings.file_format = 'PNG'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    print("✅ Render settings configured")

    bbox_corners_final = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_x = min(c.x for c in bbox_corners_final)
    max_x = max(c.x for c in bbox_corners_final)
    min_y = min(c.y for c in bbox_corners_final)
    max_y = max(c.y for c in bbox_corners_final)
    min_z = min(c.z for c in bbox_corners_final)
    max_z = max(c.z for c in bbox_corners_final)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z

    print(f"Model bounds: {{size_x:.1f}} x {{size_y:.1f}} x {{size_z:.1f}}")

    print("🎨 Creating AI-textured material...")
    
    building_mat = bpy.data.materials.new(name="AI_Facade_Material")
    building_mat.use_nodes = True
    nodes = building_mat.node_tree.nodes
    links = building_mat.node_tree.links
    nodes.clear()
    
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    ai_texture_path_str = "{ai_texture_path_str if ai_texture_path_str else ""}"

    if ai_texture_path_str and os.path.exists(ai_texture_path_str):
        print(f"🖼️ Loading AI texture from: {{ai_texture_path_str}}")
        tex_image_node = nodes.new(type='ShaderNodeTexImage')
        tex_image_node.image = bpy.data.images.load(ai_texture_path_str)
        
        tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
        mapping_node = nodes.new(type='ShaderNodeMapping')
        
        mapping_node.inputs['Scale'].default_value = (0.1, 0.1, 0.1)
        mapping_node.vector_type = 'TEXTURE'

        links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], tex_image_node.inputs['Vector'])
        links.new(tex_image_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        links.new(tex_image_node.outputs['Alpha'], bsdf_node.inputs['Roughness'])
        bsdf_node.inputs['Roughness'].default_value = 0.6
        print("✅ AI Texture applied to material.")
    else:
        print("⚠️ AI Texture path not valid or not provided. Using fallback color.")
        bsdf_node.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)

    if len(obj.material_slots) == 0:
        obj.data.materials.append(building_mat)
    else:
        obj.data.materials[0] = building_mat
    print("✅ Building material assigned.")

    print("🌍 Adding ground...")
    ground_size = max(size_x, size_y) * 2.0
    bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(center_x, center_y, min_z - 0.1))
    ground = bpy.context.active_object
    
    ground_mat = bpy.data.materials.new(name="Ground_Dark")
    ground_mat.use_nodes = True
    ground_bsdf = ground_mat.node_tree.nodes.get("Principled BSDF")
    if ground_bsdf:
        ground_bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1.0)
        ground_bsdf.inputs['Roughness'].default_value = 0.8
    ground.data.materials.append(ground_mat)

    print("💡 Setting up lighting...")
    bpy.ops.object.light_add(type='SUN', location=(center_x + size_x, center_y + size_y, center_z + size_z))
    sun = bpy.context.active_object
    sun.data.energy = 7.0
    sun.data.angle = math.radians(2)
    sun.rotation_euler = (math.radians(50), math.radians(-30), math.radians(170))

    print("🏙️ Adjusting sky for urban feel...")
    world = scene.world
    if not world:
        world = bpy.data.worlds.new("UrbanWorld")
        scene.world = world

    world.use_nodes = True
    sky_nodes = world.node_tree.nodes
    sky_links = world.node_tree.links
    sky_nodes.clear()

    sky_tex_node = sky_nodes.new(type='ShaderNodeTexSky')
    bg_node = sky_nodes.new(type='ShaderNodeBackground')
    world_output_node = sky_nodes.new(type='ShaderNodeOutputWorld')

    sky_links.new(sky_tex_node.outputs['Color'], bg_node.inputs['Color'])
    sky_links.new(bg_node.outputs['Background'], world_output_node.inputs['Surface'])

    sky_tex_node.sky_type = 'NISHITA'
    try:
        if 'Sun Elevation' in sky_tex_node.inputs:
            sky_tex_node.inputs['Sun Elevation'].default_value = math.radians(35)
        elif 'Sun Height' in sky_tex_node.inputs:
            sky_tex_node.inputs['Sun Height'].default_value = math.radians(35)
        
        if 'Sun Rotation' in sky_tex_node.inputs:
            sky_tex_node.inputs['Sun Rotation'].default_value = math.radians(160)
        
        if 'Altitude' in sky_tex_node.inputs:
            sky_tex_node.inputs['Altitude'].default_value = 100
        if 'Air Density' in sky_tex_node.inputs:
            sky_tex_node.inputs['Air Density'].default_value = 2.5
        elif 'Air' in sky_tex_node.inputs:
            sky_tex_node.inputs['Air'].default_value = 2.5
        if 'Dust Density' in sky_tex_node.inputs:
            sky_tex_node.inputs['Dust Density'].default_value = 1.5
        elif 'Dust' in sky_tex_node.inputs:
            sky_tex_node.inputs['Dust'].default_value = 1.5
        if 'Ozone Density' in sky_tex_node.inputs:
            sky_tex_node.inputs['Ozone Density'].default_value = 3.0
        elif 'Ozone' in sky_tex_node.inputs:
            sky_tex_node.inputs['Ozone'].default_value = 3.0
    except Exception as e:
        print(f"⚠️ Some sky properties not available: {{e}}")
        pass
    bg_node.inputs['Strength'].default_value = 0.8
    print("✅ Urban sky configured.")

    print("📸 Setting up cameras for 3 perspectives...")
    camera_distance = max(size_x, size_y) * 1.3
    camera_height_offset = size_z * 0.6
    
    camera_angles = [
        (30, "perspective_front_right"), 
        (150, "perspective_front_left"),  
        (270, "perspective_side_profile") 
    ]
    
    base_camera = bpy.data.cameras.new(name="MultiViewCam")
    base_camera.lens = 40
    
    cam_obj = bpy.data.objects.new(name="MultiViewCamObj", object_data=base_camera)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    
    for angle_deg, view_name in camera_angles:
        print(f"Setting up {{view_name}} ({{angle_deg}}°)...")
        angle_rad = math.radians(angle_deg)
        
        cam_x = center_x + camera_distance * math.cos(angle_rad)
        cam_y = center_y + camera_distance * math.sin(angle_rad)
        cam_z = center_z + camera_height_offset
        
        cam_obj.location = (cam_x, cam_y, cam_z)
        
        direction = Vector((center_x, center_y, center_z)) - cam_obj.location
        cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        
        output_file_path = os.path.join("{output_dir_str}", f"{{view_name}}.png")
        scene.render.filepath = output_file_path
        
        print(f"🎬 Rendering {{view_name}} to {{output_file_path}}")
        bpy.ops.render.render(write_still=True)
        print(f"✅ {{view_name}} rendered.")

    blend_path = os.path.join("{output_dir_str}", "ai_detailed_model.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"💾 Saved blend file to {{blend_path}}")

    print("✅ AI-Detailed multi-perspective rendering completed!")

except Exception as e_blender:
    print(f"❌ Error in Blender script: {{e_blender}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        return script
    
    def run_blender_script(self, script_content: str):
        """Run Blender script."""
        print("🧊 Running Blender for AI-detailed multi-perspective rendering...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            blender_cmd = ['blender', '--background', '--python', script_path, '--python-exit-code', '1']
            process = subprocess.run(
                blender_cmd, capture_output=True, text=True, timeout=720
            )
            
            if process.stdout:
                for line in process.stdout.split('\n'):
                    if line.strip() and any(m in line for m in ['🔄', '✅', '❌', '🎨', '🏢', '🌍', '💡', '📸', '🎬', '💾', '🖼️', '🏙️']):
                        print(f"  {line}")
            
            if process.returncode != 0:
                print("⚠️ Blender process failed.")
                if process.stderr:
                    print("Blender Errors:")
                    for line in process.stderr.split('\n'): 
                        if line.strip(): print(f"  {line}")
                return False
            print("✅ Blender processing completed successfully.")
            return True
                
        except Exception as e:
            print(f"❌ Error running Blender: {e}")
            return False
        finally:
            if os.path.exists(script_path): 
                os.unlink(script_path)
    
    def process_model(self, ai_texture_resolution: int = 1024):
        """Process OBJ model with AI texture and multiple perspective renders."""
        start_time = time.time()
        
        print(f"🏙️ AI-Detailed Multi-Perspective OBJ Mapper")
        print(f"   OBJ: {self.obj_path} (Exists: {self.obj_path.exists()})")
        print(f"   Output: {self.output_dir}")
        print(f"   AI Texture Res: {ai_texture_resolution}px")
        print(f"   AI Device: {self.device}")
        
        if not self.obj_path.exists():
            print(f"❌ OBJ file not found.")
            return

        # Generate AI Facade Texture First
        ai_texture_file = self.generate_facade_texture(resolution=ai_texture_resolution)
        if not ai_texture_file:
            print("❌ Proceeding without AI texture due to generation failure.")
        
        vertices, faces = self.parse_obj()
        if not vertices or not faces:
            print("❌ No valid geometry found in OBJ file.")
            return
        
        ai_texture_path_str = str(ai_texture_file.resolve()) if ai_texture_file else None
        script_content = self.create_blender_script(vertices, faces, ai_texture_path_str)
        
        if not self.run_blender_script(script_content):
            print("❌ Blender script execution failed.")
            return
        
        print(f"\n📊 Final Results (in {self.output_dir}):")
        for view_name in ["perspective_front_right", "perspective_front_left", "perspective_side_profile"]:
            pfile = self.output_dir / f"{view_name}.png"
            exists = pfile.exists()
            size = f" ({pfile.stat().st_size / 1024:.1f} KB)" if exists else ""
            print(f"   {view_name}.png: {exists}{size}")
        
        blend_file = self.output_dir / "ai_detailed_model.blend"
        blend_exists = blend_file.exists()
        blend_size = f" ({blend_file.stat().st_size / 1024:.1f} KB)" if blend_exists else ""
        print(f"   ai_detailed_model.blend: {blend_exists}{blend_size}")
        
        print(f"\n✅ Total processing completed in {(time.time() - start_time):.1f} seconds")

def main():
    parser = argparse.ArgumentParser(description="AI-Detailed Multi-Perspective OBJ Mapper")
    parser.add_argument("obj_path", type=str, help="Path to the OBJ file")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--resolution", type=int, default=512, help="AI Texture resolution (default: 512)")
    parser.add_argument("--device", type=str, default=None, help="Force AI device (cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    mapper = AiDetailedObjMapper(
        obj_path=args.obj_path,
        output_dir=args.output_dir,
        device=args.device
    )
    mapper.process_model(ai_texture_resolution=args.resolution)

if __name__ == "__main__":
    main() 