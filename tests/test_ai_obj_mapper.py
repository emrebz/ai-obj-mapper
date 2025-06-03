import unittest
from pathlib import Path
import sys
import os

# Add the parent directory to sys.path to allow importing ai_obj_mapper
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from unittest.mock import patch, MagicMock, call, mock_open # Added 'call' for mock_mkdir.assert_has_calls and mock_open

# --- Start of global torch mock ---
MOCK_TORCH_MODULE = True
if MOCK_TORCH_MODULE:
    mock_torch = MagicMock()

    mock_torch.cuda = MagicMock()
    mock_torch.cuda.is_available = MagicMock(return_value=False) # Default mock for initialization

    mock_torch.backends = MagicMock()
    mock_torch.backends.mps = MagicMock()
    # Configure MPS to be available by default for hasattr check, specific tests will override .is_available
    # hasattr(torch.backends, 'mps') will be true, then torch.backends.mps.is_available() is called.
    mock_torch.backends.mps.is_available = MagicMock(return_value=False) # Default to False, tests will specify True

    mock_torch.float16 = object()
    mock_torch.float32 = object()

    mock_no_grad_context_manager = MagicMock()
    mock_no_grad_context_manager.__enter__.return_value = None
    mock_no_grad_context_manager.__exit__.return_value = None
    # ai_obj_mapper.py uses 'with torch.no_grad():'
    mock_torch.no_grad = lambda: mock_no_grad_context_manager

    sys.modules['torch'] = mock_torch
# --- End of global torch mock ---

# Now import AiDetailedObjMapper after torch has been mocked
from ai_obj_mapper import AiDetailedObjMapper


class TestAiDetailedObjMapperInitialization(unittest.TestCase):

    @patch('ai_obj_mapper.Path.mkdir') # Mocks Path.mkdir for all instances
    def test_initialization_basic(self, mock_mkdir_method):
        """Test basic initialization of AiDetailedObjMapper."""
        obj_path = "dummy.obj"
        output_dir = "output/test_init"

        # Test with device explicitly set to 'cpu' or ensure mocks make it fall back to 'cpu'
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            mapper = AiDetailedObjMapper(obj_path=obj_path, output_dir=output_dir)

        self.assertEqual(mapper.obj_path, Path(obj_path))
        self.assertEqual(mapper.output_dir, Path(output_dir))
        self.assertEqual(mapper.texture_dir, Path(output_dir) / "ai_textures")
        self.assertEqual(mapper.device, "cpu")

        # Check that output_dir.mkdir and texture_dir.mkdir were called
        # Path.mkdir is patched, so we check calls on the mock_mkdir_method
        # mapper.output_dir and mapper.texture_dir are Path objects.
        # Their .mkdir() method will use the patched Path.mkdir.

        # We need to ensure that the *instances* of Path had their mkdir called.
        # This is tricky if Path.mkdir is a static method or if Path objects are created fresh.
        # The current mock @patch('ai_obj_mapper.Path.mkdir') mocks the class's method.
        # So, any Path(...).mkdir(...) will use this mock.

        expected_mkdir_calls = [
            call(exist_ok=True, parents=True), # For self.output_dir.mkdir()
            call(exist_ok=True, parents=True)  # For self.texture_dir.mkdir()
        ]
        mock_mkdir_method.assert_has_calls(expected_mkdir_calls, any_order=True)
        self.assertEqual(mock_mkdir_method.call_count, 2)


    @patch('ai_obj_mapper.Path.mkdir')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.backends.mps.is_available', return_value=False) # Ensure MPS is false for this test
    def test_device_selection_cuda(self, mock_mps_available, mock_cuda_available, mock_mkdir_method):
        """Test device selection prefers CUDA when available."""
        mapper = AiDetailedObjMapper(obj_path="dummy.obj", output_dir="output/cuda_test")
        self.assertEqual(mapper.device, "cuda")

    @patch('ai_obj_mapper.Path.mkdir')
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_device_selection_mps(self, mock_mps_available, mock_cuda_available, mock_mkdir_method):
        """Test device selection falls back to MPS when CUDA is not available."""
        mapper = AiDetailedObjMapper(obj_path="dummy.obj", output_dir="output/mps_test")
        self.assertEqual(mapper.device, "mps")

    @patch('ai_obj_mapper.Path.mkdir')
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_device_selection_cpu(self, mock_mps_available, mock_cuda_available, mock_mkdir_method):
        """Test device selection falls back to CPU when CUDA and MPS are not available."""
        mapper = AiDetailedObjMapper(obj_path="dummy.obj", output_dir="output/cpu_test")
        self.assertEqual(mapper.device, "cpu")

    @patch('ai_obj_mapper.Path.mkdir')
    def test_device_selection_override(self, mock_mkdir_method):
        """Test device selection can be overridden by constructor argument."""
        forced_device = "test_device"
        # Patch cuda and mps to be true to ensure the override is effective
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.backends.mps.is_available', return_value=True):
            mapper = AiDetailedObjMapper(obj_path="dummy.obj", output_dir="output/override_test", device=forced_device)
        self.assertEqual(mapper.device, forced_device)

if __name__ == '__main__':
    unittest.main()


class TestParseObj(unittest.TestCase):

    def setUp(self):
        # Basic AiDetailedObjMapper instance for testing parse_obj
        # obj_path and output_dir are nominal for this test, as 'open' will be mocked.
        # We patch mkdir to avoid side effects during this unrelated test setup.
        # Also, mock torch globally if it's not already fully mocked for AiDetailedObjMapper init
        global_torch_mock_needed = 'torch' not in sys.modules or isinstance(sys.modules['torch'], MagicMock)

        # Ensure relevant torch attributes are specifically available for AiDetailedObjMapper init if needed
        # However, the main AiDetailedObjMapper class should already be importable due to prior global mocks.
        # The __init__ for AiDetailedObjMapper determines the device. We can let it use its default (cpu).
        with patch('ai_obj_mapper.Path.mkdir'): # Mock mkdir for this setup
            self.mapper = AiDetailedObjMapper(obj_path="dummy_for_parse.obj", output_dir="dummy_output_for_parse")


    @patch('builtins.open')
    def test_parse_simple_obj(self, mock_open_file_func): # Renamed to avoid conflict with mock_open import
        """Test parsing a simple OBJ string with vertices and faces."""
        simple_obj_content = """
# A simple cube
v 1.0 1.0 -1.0
v 1.0 -1.0 -1.0
v 1.0 1.0 1.0
v 1.0 -1.0 1.0
v -1.0 1.0 -1.0
v -1.0 -1.0 -1.0
v -1.0 1.0 1.0
v -1.0 -1.0 1.0

f 1//1 2//1 4//1 3//1
f 5//2 6//2 8//2 7//2
f 1//3 5//3 7//3 3//3
f 2//4 6//4 8//4 4//4
f 3//5 7//5 8//5 4//5
f 1//6 5//6 6//6 2//6
"""
        mock_open_file_func.return_value = mock_open(read_data=simple_obj_content).return_value

        vertices, faces = self.mapper.parse_obj()

        expected_vertices = [
            [1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]
        ]
        expected_faces = [
            [0, 1, 3, 2], [4, 5, 7, 6], [0, 4, 6, 2],
            [1, 5, 7, 3], [2, 6, 7, 3], [0, 4, 5, 1]
        ]

        self.assertEqual(vertices, expected_vertices)
        self.assertEqual(faces, expected_faces)
        mock_open_file_func.assert_called_once_with(self.mapper.obj_path, 'r')

    @patch('builtins.open')
    def test_parse_obj_with_comments_and_empty_lines(self, mock_open_file_func):
        """Test parsing OBJ with comments, empty lines, and different face formats."""
        obj_content = """
# This is a comment
v 1.0 0.0 0.0
v 0.0 1.0 0.0

v 0.0 0.0 1.0 # Another comment

f 1 2 3
f 3/1 2/2 1/3
f 3//1 2//2 1//3
"""
        mock_open_file_func.return_value = mock_open(read_data=obj_content).return_value

        vertices, faces = self.mapper.parse_obj()

        expected_vertices = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        expected_faces = [[0, 1, 2], [2, 1, 0], [2, 1, 0]]

        self.assertEqual(vertices, expected_vertices)
        self.assertEqual(faces, expected_faces)

    @patch('builtins.open')
    def test_parse_empty_obj(self, mock_open_file_func):
        """Test parsing an empty OBJ string."""
        mock_open_file_func.return_value = mock_open(read_data="").return_value

        vertices, faces = self.mapper.parse_obj()

        self.assertEqual(vertices, [])
        self.assertEqual(faces, [])

    @patch('builtins.open')
    def test_parse_obj_no_faces(self, mock_open_file_func):
        """Test parsing OBJ with vertices but no faces."""
        obj_content = """
v 1.0 0.0 0.0
v 0.0 1.0 0.0
"""
        mock_open_file_func.return_value = mock_open(read_data=obj_content).return_value

        vertices, faces = self.mapper.parse_obj()

        expected_vertices = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        self.assertEqual(vertices, expected_vertices)
        self.assertEqual(faces, [])

    @patch('builtins.open')
    def test_parse_obj_no_vertices(self, mock_open_file_func):
        """Test parsing OBJ with faces but no vertices (parsed as empty)."""
        obj_content = "f 1 2 3" # Invalid OBJ, faces refer to non-existent vertices
        mock_open_file_func.return_value = mock_open(read_data=obj_content).return_value

        vertices, faces = self.mapper.parse_obj()

        self.assertEqual(vertices, [])
        self.assertEqual(faces, [])


class TestCreateBlenderScript(unittest.TestCase):

    def setUp(self):
        # Basic AiDetailedObjMapper instance for testing create_blender_script
        # obj_path is nominal. output_dir is used by the method.
        # Patch mkdir to avoid side effects.
        with patch('ai_obj_mapper.Path.mkdir'):
            self.output_dir_path = Path("output/test_blender_script")
            self.mapper = AiDetailedObjMapper(obj_path="dummy_for_script.obj", output_dir=str(self.output_dir_path))

    def test_script_generation_with_texture(self):
        """Test Blender script generation when an AI texture path is provided."""
        vertices = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        faces = [[0, 1, 2]]
        ai_texture_path_str = "/fake/path/to/ai_texture.png"

        # Mock os.path.exists for the texture path check within create_blender_script
        # The script uses 'if ai_texture_path_str and os.path.exists(ai_texture_path_str):'
        with patch('os.path.exists', return_value=True) as mock_exists:
            script = self.mapper.create_blender_script(vertices, faces, ai_texture_path_str)
            mock_exists.assert_called_once_with(ai_texture_path_str)

        self.assertIsInstance(script, str)
        self.assertIn(f"vertices = {str(vertices)}", script)
        self.assertIn(f"faces = {str(faces)}", script)
        self.assertIn(f'ai_texture_path_str = "{ai_texture_path_str}"', script)
        self.assertIn("tex_image_node.image = bpy.data.images.load(ai_texture_path_str)", script)
        self.assertIn(f"output_dir_str = str(Path('{str(self.output_dir_path.resolve())}'))", script.replace("\\\\", "/").replace("\\", "/")) # Normalize path for comparison
        self.assertIn("perspective_front_right.png", script)
        self.assertIn("perspective_front_left.png", script)
        self.assertIn("perspective_side_profile.png", script)
        self.assertIn("bpy.ops.wm.save_as_mainfile", script)
        self.assertIn("Cycles render settings check", self._get_cycles_settings_check_snippet(), script)


    def test_script_generation_without_texture(self):
        """Test Blender script generation when no AI texture path is provided (None)."""
        vertices = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        faces = [[0, 1, 2]]
        ai_texture_path_str = None # No texture

        # os.path.exists should not be called if ai_texture_path_str is None
        with patch('os.path.exists') as mock_exists:
            script = self.mapper.create_blender_script(vertices, faces, ai_texture_path_str)
            mock_exists.assert_not_called()

        self.assertIsInstance(script, str)
        self.assertIn(f"vertices = {str(vertices)}", script)
        self.assertIn(f"faces = {str(faces)}", script)
        # Check that the placeholder for texture path is empty or correctly handled
        self.assertIn('ai_texture_path_str = ""', script)
        self.assertIn("print(\"⚠️ AI Texture path not valid or not provided. Using fallback color.\")", script)
        self.assertNotIn("tex_image_node.image = bpy.data.images.load(ai_texture_path_str)", script)
        self.assertIn("bsdf_node.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)", script) # Fallback color
        self.assertIn(f"output_dir_str = str(Path('{str(self.output_dir_path.resolve())}'))", script.replace("\\\\", "/").replace("\\", "/"))
        self.assertIn("Cycles render settings check", self._get_cycles_settings_check_snippet(), script)


    def test_script_generation_with_invalid_texture_path(self):
        """Test script generation with a texture path that os.path.exists returns False for."""
        vertices = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        faces = [[0, 1, 2]]
        ai_texture_path_str = "/fake/path/to/non_existent_texture.png"

        with patch('os.path.exists', return_value=False) as mock_exists: # Texture path does not exist
            script = self.mapper.create_blender_script(vertices, faces, ai_texture_path_str)
            mock_exists.assert_called_once_with(ai_texture_path_str)

        self.assertIn('ai_texture_path_str = "{}"'.format(ai_texture_path_str), script) # Path is still passed
        self.assertIn("print(\"⚠️ AI Texture path not valid or not provided. Using fallback color.\")", script)
        self.assertNotIn("tex_image_node.image = bpy.data.images.load(ai_texture_path_str)", script) # But not loaded
        self.assertIn("bsdf_node.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)", script)

    def _get_cycles_settings_check_snippet(self):
        # Helper to avoid repeating this snippet, makes tests cleaner
        # This is just a placeholder for a more specific check if needed
        return "bpy.context.scene.render.engine = 'CYCLES'"


class TestProcessModelTextureLogic(unittest.TestCase):

    def setUp(self):
        # We need an instance of AiDetailedObjMapper.
        # Mock Path.mkdir to prevent actual directory creation during its __init__.
        # Mock obj_path.exists() to return True for the initial check in process_model.
        self.obj_file_path = Path("test_model.obj")
        self.output_dir_path = Path("output/test_process_model")

        with patch('ai_obj_mapper.Path.mkdir'),              patch.object(Path, 'exists', return_value=True) as mock_path_exists:
            self.mapper = AiDetailedObjMapper(obj_path=str(self.obj_file_path), output_dir=str(self.output_dir_path))
            # Check that Path.exists was called for the obj_path in __init__ (if it is) or in process_model
            # It is called in process_model: if not self.obj_path.exists():
            # So, this setup mock_path_exists will cover that.

    @patch.object(AiDetailedObjMapper, 'run_blender_script', return_value=True) # Mock to prevent execution
    @patch.object(AiDetailedObjMapper, 'create_blender_script', return_value="mocked_script") # Mock to check calls
    @patch.object(AiDetailedObjMapper, 'parse_obj', return_value=([[1,0,0]], [[0,0,0]])) # Mock to provide dummy data
    @patch.object(AiDetailedObjMapper, 'generate_facade_texture') # Mock this core method
    @patch.object(Path, 'exists') # Mock for self.obj_path.exists() within process_model
    def test_process_model_with_generated_texture(
            self, mock_path_exists, mock_generate_texture, mock_parse_obj,
            mock_create_script, mock_run_script):
        """Test process_model calls create_blender_script with texture path if generated."""
        mock_path_exists.return_value = True # For self.obj_path.exists()

        fake_texture_path = self.mapper.texture_dir / "ai_facade_texture.png"
        mock_generate_texture.return_value = fake_texture_path # Simulate successful texture generation

        self.mapper.process_model(ai_texture_resolution=64) # Low res for speed if it were real

        mock_generate_texture.assert_called_once_with(resolution=64)
        mock_parse_obj.assert_called_once()
        mock_create_script.assert_called_once_with(
            mock_parse_obj.return_value[0], # vertices
            mock_parse_obj.return_value[1], # faces
            str(fake_texture_path.resolve()) # Expected texture path string
        )
        mock_run_script.assert_called_once_with("mocked_script")

    @patch.object(AiDetailedObjMapper, 'run_blender_script', return_value=True)
    @patch.object(AiDetailedObjMapper, 'create_blender_script', return_value="mocked_script")
    @patch.object(AiDetailedObjMapper, 'parse_obj', return_value=([[1,0,0]], [[0,0,0]]))
    @patch.object(AiDetailedObjMapper, 'generate_facade_texture')
    @patch.object(Path, 'exists')
    def test_process_model_without_generated_texture(
            self, mock_path_exists, mock_generate_texture, mock_parse_obj,
            mock_create_script, mock_run_script):
        """Test process_model calls create_blender_script with None if texture fails."""
        mock_path_exists.return_value = True # For self.obj_path.exists()

        mock_generate_texture.return_value = None # Simulate failed texture generation

        self.mapper.process_model(ai_texture_resolution=64)

        mock_generate_texture.assert_called_once_with(resolution=64)
        mock_parse_obj.assert_called_once()
        mock_create_script.assert_called_once_with(
            mock_parse_obj.return_value[0], # vertices
            mock_parse_obj.return_value[1], # faces
            None # Expected texture path is None
        )
        mock_run_script.assert_called_once_with("mocked_script")

    @patch.object(Path, 'exists', return_value=False) # Mock for self.obj_path.exists()
    @patch.object(AiDetailedObjMapper, 'generate_facade_texture')
    def test_process_model_obj_not_found(self, mock_generate_texture, mock_path_exists):
        """Test process_model exits early if OBJ file does not exist."""
        # Re-initialize mapper for this specific scenario if needed, or ensure obj_path is correctly set
        # The setUp mock_path_exists might interfere if not careful.
        # Here, the @patch.object on the test method takes precedence for Path.exists.

        self.mapper.process_model()

        mock_path_exists.assert_called_with(self.mapper.obj_path) # Check it was called for the obj file
        mock_generate_texture.assert_not_called() # Should not proceed to texture generation

    @patch.object(AiDetailedObjMapper, 'run_blender_script')
    @patch.object(AiDetailedObjMapper, 'create_blender_script')
    @patch.object(AiDetailedObjMapper, 'parse_obj', return_value=([], [])) # No geometry
    @patch.object(AiDetailedObjMapper, 'generate_facade_texture', return_value=Path("dummy_texture.png"))
    @patch.object(Path, 'exists', return_value=True) # OBJ and texture exist
    def test_process_model_no_geometry(
            self, mock_path_exists, mock_generate_texture, mock_parse_obj,
            mock_create_script, mock_run_script):
        """Test process_model exits if OBJ parsing yields no geometry."""

        self.mapper.process_model()

        mock_generate_texture.assert_called_once() # Texture generation still happens
        mock_parse_obj.assert_called_once()
        mock_create_script.assert_not_called() # Should not proceed to script creation
        mock_run_script.assert_not_called()
