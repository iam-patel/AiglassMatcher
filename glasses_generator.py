import warnings
warnings.filterwarnings("ignore")

import os
import io
import cv2
import numpy as np
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
if not HF_API_KEY:
    print("[glasses] ERROR: HF_API_KEY not found in .env file!")
    print("[glasses] Get free key from: https://huggingface.co/settings/tokens")
    client = None
else:
    client = "ready"

# Hugging Face Stable Diffusion Inpainting model endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-inpainting"

STYLE_PROMPTS = {
   
    "round":    "Add black round eyeglasses on the person's face. High quality, realistic, professional.",
    "square":   "Add square frame eyeglasses on the person's face. High quality, realistic, professional.",
    "wayfarer": "Add wayfarer style sunglasses on the person's face. High quality, realistic, professional.",
    "oval":     "Add oval frame eyeglasses on the person's face. High quality, realistic, professional.",
}

STYLE_CYCLE = [ "round", "square", "wayfarer", "oval"]
_try_count = 0


def generate_glasses_image(face_image_bgr, style: str = "glasses", face_shape: str = "oval", try_count: int = 0):
    """
    Uses Hugging Face Stable Diffusion to add glasses to user's face.
    Returns (result_bgr, message, style_used)
    """
    global _try_count
    
    if client is None:
        raise RuntimeError("HF_API_KEY not configured. See .env file.")

    # Pick style from cycle
    style_to_use = STYLE_CYCLE[try_count % len(STYLE_CYCLE)]
    prompt = STYLE_PROMPTS.get(style_to_use, STYLE_PROMPTS["wayfarer"])

    print(f"[HF] Sending to Hugging Face... style={style_to_use}")

    # Convert image to PNG bytes
    _, img_bytes = cv2.imencode(".png", face_image_bgr)
    
    # Create mask (full image white = edit everywhere)
    h, w = face_image_bgr.shape[:2]
    mask = np.ones((h, w, 3), dtype=np.uint8) * 255
    _, mask_bytes = cv2.imencode(".png", mask)

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    files = {
        "inputs": img_bytes.tobytes(),
    }
    
    data = {
        "prompt": prompt,
    }

    try:
        # Send to Hugging Face
        response = requests.post(
            HF_API_URL,
            headers=headers,
            files={"image": ("image.png", img_bytes.tobytes(), "image/png")},
            data={"inputs": prompt},
            timeout=60
        )

        if response.status_code != 200:
            print(f"[HF] Error: {response.status_code}")
            print(f"[HF] Response: {response.text[:200]}")
            raise RuntimeError(f"Hugging Face API error: {response.status_code}")

        # Parse response
        result_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        result_bgr = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

        message = f"{style_to_use.capitalize()} frames suit your {face_shape} face perfectly!"
        print(f"[HF] Success! Generated {style_to_use} glasses")
        
        return result_bgr, message, style_to_use

    except requests.exceptions.Timeout:
        raise RuntimeError("Hugging Face request timed out. Try again.")
    except Exception as e:
        raise RuntimeError(f"Hugging Face generation failed: {e}")