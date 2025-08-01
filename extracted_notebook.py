# --- Cell 0 ---
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# --- Cell 1 ---
!pip install -U bitsandbytes
!pip install faiss-cpu sentence-transformers
!pip install gradio

# --- Cell 2 ---
!pip install diffusers transformers accelerate safetensors
!pip install gtts


# --- Cell 3 ---
!pip install -q transformers accelerate peft
!pip install PyPDF2
!pip install pyttsx3

# --- Cell 4 ---
# === Imports ===
import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from PyPDF2 import PdfReader
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import zipfile
import os
import uuid
import tempfile

# --- Cell 5 ---
import nltk
nltk.download('punkt')
print(torch.cuda.is_available())

# --- Cell 6 ---
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")

# --- Cell 7 ---
# # === Load Base Model and LoRA Adapter ===
# base_model = "meta-llama/Llama-3.2-3B-Instruct"
# adapter_path = "/kaggle/input/lora-finetuned-mood-story"

# tokenizer = AutoTokenizer.from_pretrained(base_model)
# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     device_map="auto",
#     torch_dtype=torch.float16
# )
# model = PeftModel.from_pretrained(model, adapter_path)
# model.eval()

# --- Cell 8 ---
# # === Embedding Model for RAG ===
# embedder = SentenceTransformer("all-MiniLM-L6-v2")


# --- Cell 9 ---
# # === Lore Processing Functions ===
# def load_custom_lore(file):
#     if file is None:
#         return ""
#     try:
#         if file.name.endswith('.pdf'):
#             reader = PdfReader(file)
#             return "\n".join([page.extract_text() or "" for page in reader.pages])
#         else:
#             with open(file.name, 'r', encoding='utf-8') as f:
#                 return f.read()
#     except Exception as e:
#         return f"[Error reading file: {e}]"

# def chunk_text(text, chunk_size=100):
#     words = text.split()
#     return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# --- Cell 10 ---
# # === Story Generation ===
# def generate_rag_story(prompt, mood, rag_enabled, lore_data):
#     index, chunks = lore_data if lore_data else (None, None)

#     if rag_enabled and index is not None and chunks:
#         context = retrieve_context(prompt, index, chunks)
#     else:
#         context = ""

#     final_prompt = f"[Mood: {mood}]\nContext: {context}\nPrompt: {prompt}"

#     input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to(model.device)
#     output_ids = model.generate(input_ids, max_new_tokens=300, do_sample=True, temperature=0.7)
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# --- Cell 11 ---
# # === Gradio UI ===
# def process_lore(file):
#     text = load_custom_lore(file)
#     if text.startswith("[Error"):
#         return None, None
#     chunks = chunk_text(text)
#     index, _ = build_faiss_index(chunks)
#     return index, chunks

# with gr.Blocks() as demo:
#     with gr.Row():
#         prompt = gr.Textbox(label="Enter Your Prompt")
#         mood = gr.Textbox(label="Mood (e.g., happy, sad, mysterious)")

#     with gr.Row():
#         rag_enabled = gr.Checkbox(label="📚 Use Uploaded Lore (RAG)", value=False)
#         lore_upload = gr.File(label="Upload Your Story Universe (.pdf or .txt)", file_types=[".pdf", ".txt"])
#         lore_state = gr.State()

#     generated_output = gr.Textbox(label="Generated Story", lines=10)
#     generate_btn = gr.Button("Generate Story")

#     lore_upload.change(fn=process_lore, inputs=lore_upload, outputs=lore_state)
#     generate_btn.click(fn=generate_rag_story, inputs=[prompt, mood, rag_enabled, lore_state], outputs=generated_output)

# demo.launch()

# --- Cell 12 ---
# === Imports ===
import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from PyPDF2 import PdfReader
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import zipfile
import os
import uuid
import tempfile
from PIL import Image, ImageDraw

# === Load Base Model and LoRA Adapter ===
base_model = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "/kaggle/input/lora-finetuned-mood-story"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# === Stable Diffusion ===
try:
    sd_model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    print(f"[Stable Diffusion Load Error]: {e}")
    sd_model = None

# === Embedding Model for RAG ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Lore Processing Functions ===
def load_custom_lore(file):
    """Load text from PDF or TXT file."""
    if file is None:
        return "[No lore uploaded]"
    try:
        if file.name.endswith('.pdf'):
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if not text.strip():
                return "[Error: PDF has no extractable text]"
            return text
        elif file.name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return "[Error: Unsupported file type]"
    except Exception as e:
        return f"[Error reading file: {e}]"

def chunk_text(text, chunk_size=100):
    """Split text into chunks for FAISS indexing."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(chunks):
    """Create FAISS index from text chunks."""
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

def retrieve_context(prompt, index, chunks, k=3):
    """Retrieve top-k relevant lore chunks for a given prompt."""
    query_embedding = embedder.encode([prompt])
    distances, indices = index.search(np.array(query_embedding), k)
    return "\n".join([chunks[i] for i in indices[0]])

# === Generation Functions with History ===
story_history = []

genres = ["Fantasy", "Sci-Fi", "Mystery", "Adventure", "Romance", "Horror", "Comedy", "Drama"]

def generate_story_with_history(prompt, genre, mood, history, lore_data, rag_enabled):
    """Generate story continuation or start fresh."""
    if lore_data and isinstance(lore_data, tuple) and len(lore_data) == 2:
        index, chunks = lore_data
    else:
        index, chunks = None, None

    context = ""
    if rag_enabled and index is not None and chunks:
        context = retrieve_context(prompt, index, chunks)

    # Compose prompt for LLaMA
    if history and prompt.strip() == history[-1]["prompt"].strip():
        prev_story = history[-1]["story"]
        final_prompt = (
            f"[Genre: {genre}] [Mood: {mood}]\n{prompt}\n\n"
            f"Previous Story:\n{prev_story}\n\n"
            f"Lore Context:\n{context}\n\nContinue the narrative:"
        )
    else:
        history = []
        final_prompt = (
            f"[Genre: {genre}] [Mood: {mood}]\n{prompt}\n\n"
            f"Lore Context:\n{context}\n\nStart the story:"
        )

    # Generate story
    input_ids = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_new_tokens=300, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    new_story = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Generate image
    sentences = new_story.split('.')
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

    image_path = f"/kaggle/working/story_image_{len(history)}.png"
    image.save(image_path)

    # Update history
    history.append({
        "prompt": prompt,
        "genre": genre,
        "mood": mood,
        "story": new_story,
        "image": image_path
    })

    chat_history = [(f"{entry['prompt']} (Genre: {entry['genre']}, Mood: {entry['mood']})", entry["story"]) for entry in history]
    gallery = [entry["image"] for entry in history]

    return history, chat_history, gallery

def clear_history():
    return [], [], []

def generate_narration(history):
    if not history:
        return None
    latest_story = history[-1]["story"]
    tts = gTTS(text=latest_story)
    audio_path = f"/kaggle/working/story_audio_{uuid.uuid4()}.mp3"
    tts.save(audio_path)
    return audio_path

def download_zip(history):
    if not history:
        return None
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "story_bundle.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for idx, entry in enumerate(history):
            story_txt = os.path.join(temp_dir, f"story_segment_{idx+1}.txt")
            with open(story_txt, "w", encoding="utf-8") as f:
                f.write(entry["story"])
            zipf.write(story_txt, f"story_segment_{idx+1}.txt")
            zipf.write(entry["image"], os.path.basename(entry["image"]))
    return zip_path


# --- Cell 13 ---
# === Gradio UI ===
with gr.Blocks(title="📚 Mood-Based Story Generator with RAG") as app:
    gr.Markdown("## 🎭 Mood-Based Story Generator with Images, Narration & Lore Upload")
    gr.Markdown("Enter a prompt, select a genre and mood, optionally upload lore, and click 'Generate/Continue'.")

    history_state = gr.State([])
    lore_state = gr.State()

    with gr.Row():
        prompt_input = gr.Textbox(lines=3, label="Enter your story prompt")
        genre_input = gr.Dropdown(choices=genres, label="Select Genre")
        mood_input = gr.Textbox(label="Enter Mood (e.g., dark, hopeful, mysterious)")
    
    with gr.Row():
        lore_upload = gr.File(label="📚 Upload Lore (.pdf or .txt)", file_types=[".pdf", ".txt"])
        rag_enabled = gr.Checkbox(label="Enable RAG (Use Uploaded Lore)", value=True)

    with gr.Row():
        gen_btn = gr.Button("Generate / Continue Story")
        clear_btn = gr.Button("Clear Story")

    with gr.Row():
        audio_btn = gr.Button("🎧 Narrate Latest Story")
        download_btn = gr.Button("⬇️ Download Story + Images")

    chat_output = gr.Chatbot(label="📖 Story History")
    gallery_output = gr.Gallery(label="🖼️ Story Images", columns=2)
    audio_output = gr.Audio(label="🔊 Voice Narration", type="filepath")
    download_output = gr.File(label="📦 Download Zip")

    lore_upload.change(fn=process_lore, inputs=lore_upload, outputs=lore_state)
    gen_btn.click(
        fn=generate_story_with_history,
        inputs=[prompt_input, genre_input, mood_input, history_state, lore_state, rag_enabled],
        outputs=[history_state, chat_output, gallery_output]
    )

    clear_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[history_state, chat_output, gallery_output]
    )

    audio_btn.click(
        fn=generate_narration,
        inputs=[history_state],
        outputs=[audio_output]
    )

    download_btn.click(
        fn=download_zip,
        inputs=[history_state],
        outputs=[download_output]
    )

app.launch()

# --- Cell 14 ---


