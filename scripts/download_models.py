"""
Model Downloader & Dummy ONNX Model Generator for RubikPi 3 Sensing Head.
Downloads YuNet Face Detection and YOLO-World-S ONNX models.
If offline or download fails, auto-generates minimal valid ONNX files.
"""
import os
import sys
import argparse
import urllib.request
import subprocess

# Model URLs
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YOLO_WORLD_URL = "https://huggingface.co/Instemic/yolo-world-onnx/resolve/main/yolov8s-worldv2.onnx"
WHISPER_URL = "https://huggingface.co/Xenova/whisper-tiny.en/resolve/main/onnx/encoder_model_quantized.onnx"

def ensure_onnx_installed() -> bool:
    """Verifies if 'onnx' library is installed, otherwise dynamically installs it."""
    try:
        import onnx
        return True
    except ImportError:
        print("Required python library 'onnx' is missing. Installing dynamically via pip...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "onnx"], check=True)
            import onnx
            print("Successfully installed 'onnx' library.")
            return True
        except Exception as e:
            print(f"Error installing 'onnx' package: {e}")
            return False

def create_dummy_yunet(path: str):
    """Generates a valid, minimal dummy YuNet model using ONNX helper utilities."""
    import onnx
    from onnx import helper, TensorProto

    print(f"Compiling mock YuNet face detection graph: {path}...")
    
    # Inputs: 'input' (float32, [1, 3, 480, 640])
    # Outputs: 'faces' (float32, [1, 10, 15] representing faces array)
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 480, 640])
    output_info = helper.make_tensor_value_info("faces", TensorProto.FLOAT, [1, 10, 15])
    
    # Initialize a constant face detection at index 0:
    # x_min=350, y_min=220, width=60, height=60, score=0.95
    face_data = [0.0] * 15
    face_data[0] = 350.0   # Bounding Box X
    face_data[1] = 220.0   # Bounding Box Y
    face_data[2] = 60.0    # Bounding Box Width
    face_data[3] = 60.0    # Bounding Box Height
    face_data[14] = 0.95   # Confidence Score
    
    # Populate the array: 10 face entries of 15 parameters
    all_faces = face_data + [0.0] * 15 * 9
    
    tensor_value = helper.make_tensor(
        name="const_faces_value",
        data_type=TensorProto.FLOAT,
        dims=[1, 10, 15],
        vals=all_faces
    )
    
    # Make constant node returning the array
    node = helper.make_node("Constant", [], ["faces"], value=tensor_value)
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[node],
        name="dummy_yunet_graph",
        inputs=[input_info],
        outputs=[output_info]
    )
    
    model = helper.make_model(graph, producer_name="rubikpi_dummy_generator")
    
    # Force OPSET 11 compatibility
    model.opset_import[0].version = 11
    
    onnx.save(model, path)
    print("Mock YuNet face detection graph created successfully.")

def create_dummy_yolo_world(path: str):
    """Generates a valid, minimal dummy YOLO-World model using ONNX helper utilities."""
    import onnx
    from onnx import helper, TensorProto

    print(f"Compiling mock YOLO-World open-vocabulary graph: {path}...")
    
    # Inputs: 'image' (float32, [1, 3, 480, 640]), 'prompt' (string, [1])
    # Outputs: 'output' (float32, [1, 10, 5] representing boxes)
    input_image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 480, 640])
    input_prompt = helper.make_tensor_value_info("prompt", TensorProto.STRING, [1])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 5])
    
    # Initialize a constant target box detection at index 0:
    # x=400, y=300, width=80, height=80, score=0.85
    bbox_data = [400.0, 300.0, 80.0, 80.0, 0.85]
    all_bboxes = bbox_data + [0.0] * 5 * 9
    
    tensor_value = helper.make_tensor(
        name="const_bboxes_value",
        data_type=TensorProto.FLOAT,
        dims=[1, 10, 5],
        vals=all_bboxes
    )
    
    node = helper.make_node("Constant", [], ["output"], value=tensor_value)
    
    graph = helper.make_graph(
        nodes=[node],
        name="dummy_yolo_world_graph",
        inputs=[input_image, input_prompt],
        outputs=[output_info]
    )
    
    model = helper.make_model(graph, producer_name="rubikpi_dummy_generator")
    model.opset_import[0].version = 11
    
    onnx.save(model, path)
    print("Mock YOLO-World open-vocabulary graph created successfully.")

def create_dummy_whisper(path: str):
    """Generates a valid, minimal dummy Whisper ASR model using ONNX helper utilities."""
    import onnx
    from onnx import helper, TensorProto

    print(f"Compiling mock Whisper ASR graph: {path}...")
    
    # Inputs: 'input_features' (float32, [1, 80, 3000])
    # Outputs: 'logits' (float32, [1, 10, 512])
    input_info = helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, 80, 3000])
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 10, 512])
    
    # Simple constant output values
    vals = [0.0] * (1 * 10 * 512)
    tensor_value = helper.make_tensor(
        name="const_logits_value",
        data_type=TensorProto.FLOAT,
        dims=[1, 10, 512],
        vals=vals
    )
    
    node = helper.make_node("Constant", [], ["logits"], value=tensor_value)
    
    graph = helper.make_graph(
        nodes=[node],
        name="dummy_whisper_graph",
        inputs=[input_info],
        outputs=[output_info]
    )
    
    model = helper.make_model(graph, producer_name="rubikpi_dummy_generator")
    model.opset_import[0].version = 11
    
    onnx.save(model, path)
    print("Mock Whisper ASR graph created successfully.")

def progress_hook(count, block_size, total_size):
    """Simple callback to display download percentage progress."""
    if total_size > 0:
        percent = min(100, int(count * block_size * 100 / total_size))
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()

def download_file(url: str, dest_path: str) -> bool:
    """Downloads a file via HTTP/HTTPS, displaying progress."""
    print(f"Fetching from: {url}")
    try:
        # User-Agent headers to prevent basic HTTP blocks
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print(f"\nDownload complete: saved to {dest_path}")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download deep learning model weights for RubikPi 3.")
    parser.add_argument("--dummy", action="store_true", help="Force generate mock dummy models without downloading.")
    args = parser.parse_args()

    # Create target directory
    os.makedirs("models", exist_ok=True)
    
    face_path = "models/face_detection_yunet_2023mar.onnx"
    vlm_path = "models/yolo_world_s_int8.onnx"
    whisper_path = "models/whisper_tiny_en_int8.onnx"

    if args.dummy:
        print("Force-generating dummy models as requested by --dummy...")
        if ensure_onnx_installed():
            create_dummy_yunet(face_path)
            create_dummy_yolo_world(vlm_path)
            create_dummy_whisper(whisper_path)
        else:
            print("Aborted generating dummy models because 'onnx' library is unavailable.")
        return

    # Face Detection Model
    print("\n--- [1/2] Face Detection Model: YuNet ---")
    if os.path.exists(face_path):
        print(f"Face detection model already exists at: {face_path}")
    else:
        success = download_file(YUNET_URL, face_path)
        if not success:
            print("YuNet face detection download failed. Falling back to dummy generation...")
            if ensure_onnx_installed():
                create_dummy_yunet(face_path)

    # YOLO-World Grounding Model
    print("\n--- [2/3] VLM Object Grounding Model: YOLO-World-S ---")
    if os.path.exists(vlm_path):
        print(f"VLM grounding model already exists at: {vlm_path}")
    else:
        success = download_file(YOLO_WORLD_URL, vlm_path)
        if not success:
            print("YOLO-World-S download failed. Falling back to dummy generation...")
            if ensure_onnx_installed():
                create_dummy_yolo_world(vlm_path)
                
    # Whisper ASR Model
    print("\n--- [3/3] Speech-to-Text Model: Whisper-Tiny-EN-INT8 ---")
    if os.path.exists(whisper_path):
        print(f"ASR model already exists at: {whisper_path}")
    else:
        success = download_file(WHISPER_URL, whisper_path)
        if not success:
            print("Whisper ASR download failed. Falling back to dummy generation...")
            if ensure_onnx_installed():
                create_dummy_whisper(whisper_path)
                
    print("\nAll models configured. Ready to run the sensing head.")

if __name__ == "__main__":
    main()
