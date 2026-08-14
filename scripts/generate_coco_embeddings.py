"""
Generate CLIP text embeddings for 80 COCO classes and save them to a local file.
Installs transformers and torch if not present, generates the embeddings, and saves models/coco_embeddings.npy.
Optimized to load CLIP weights once outside the loop.
"""
import os
import sys
import subprocess

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def main():
    print("Ensuring dependencies 'torch' and 'transformers' are installed...")
    try:
        import torch
        from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    except ImportError:
        print("Installing torch and transformers dynamically via pip...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "transformers", "--extra-index-url", "https://download.pytorch.org/whl/cpu"], check=True)
            import torch
            from transformers import CLIPTokenizer, CLIPTextModelWithProjection
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)

    print("Loading CLIP ViT-B/32 model...")
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model_proj = CLIPTextModelWithProjection.from_pretrained(model_id)

    print("Generating text embeddings for 80 COCO classes...")
    os.makedirs("models", exist_ok=True)
    embeddings = {}

    with torch.no_grad():
        for cls in COCO_CLASSES:
            inputs = tokenizer([cls], padding=True, return_tensors="pt")
            outputs = model_proj(**inputs)
            text_embeds = outputs.text_embeds  # shape: [1, 512]
            
            # Normalize the embedding to unit length (standard for CLIP cosine similarity)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            embeddings[cls] = text_embeds.cpu().numpy()[0]
            print(f"  Generated embedding for: '{cls}'")

    # Save to numpy format
    import numpy as np
    dest_path = "models/coco_embeddings.npy"
    np.save(dest_path, embeddings)
    print(f"\nSuccessfully compiled and saved 80 COCO class embeddings to {dest_path}")

if __name__ == "__main__":
    main()
