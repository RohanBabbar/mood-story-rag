import os
import uuid
import torch
from gtts import gTTS
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline

# === Stable Diffusion Initialization ===
try:
    sd_model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    print(f"[Stable Diffusion Load Error]: {e}")
    sd_model = None

def generate_image(prompt, history_length=0):
    """Generate an image using Stable Diffusion based on the text prompt."""
    # Use the first two sentences for image generation to avoid confusing the model
    sentences = prompt.split('.')
    image_prompt = '. '.join(sentences[:2]).strip() if sentences else prompt
    
    if sd_model:
        try:
            image = sd_model(image_prompt).images[0]
        except Exception as e:
            print(f"[Image Generation Error]: {e}")
            image = Image.new('RGB', (512, 512), color='gray')
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), "Image generation failed", fill='white')
    else:
        image = Image.new('RGB', (512, 512), color='gray')
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "No SD model loaded", fill='white')

    # Save to a temporary file instead of Kaggle working directory
    # Using /tmp ensures we don't litter the root directory
    image_path = f"/tmp/story_image_{history_length}_{uuid.uuid4().hex[:6]}.png"
    image.save(image_path)
    return image_path

def generate_narration(history):
    """Generate TTS narration for the latest story segment."""
    if not history:
        return None
    latest_story = history[-1]["story"]
    tts = gTTS(text=latest_story)
    audio_path = f"/tmp/story_audio_{uuid.uuid4().hex}.mp3"
    tts.save(audio_path)
    return audio_path
