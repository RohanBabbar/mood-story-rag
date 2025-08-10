import os
import zipfile
import tempfile
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Local Imports
from app.utils import load_custom_lore, chunk_text
from app.rag_pipeline import build_faiss_index, retrieve_context
from app.multimodal import generate_image, generate_narration

# === Load Base Model and LoRA Adapter ===
base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "/kaggle/input/lora-finetuned-mood-story" # Default Kaggle path

try:
    print("Loading LLM components...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    # Attempt to load LoRA adapter if it exists
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"Warning: LoRA adapter not found at {adapter_path}. Running base model.")
    
    model.eval()
except Exception as e:
    print(f"[Model Load Error]: {e}")
    model = None
    tokenizer = None

genres = ["Fantasy", "Sci-Fi", "Mystery", "Adventure", "Romance", "Horror", "Comedy", "Drama"]

def process_lore(file):
    """Callback to process uploaded lore document."""
    text = load_custom_lore(file)
    if text.startswith("[Error") or text.startswith("[No lore"):
        return None, None
    chunks = chunk_text(text)
    index, _ = build_faiss_index(chunks)
    return index, chunks

def generate_story_with_history(prompt, genre, mood, history, lore_data, rag_enabled):
    """Main generation loop orchestrating RAG, LLM, and Image generation."""
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

    # Generate story text
    if model and tokenizer:
        input_ids = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
        output_ids = model.generate(input_ids, max_new_tokens=300, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        new_story = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    else:
        new_story = f"[Mock Story] LLaMA models not fully loaded. Received prompt: {prompt}"

    # Generate image
    image_path = generate_image(new_story, len(history))

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
            if os.path.exists(entry["image"]):
                zipf.write(entry["image"], os.path.basename(entry["image"]))
    return zip_path

# === Gradio UI Initialization ===
def create_app():
    with gr.Blocks(title="📚 Mood-Based Story Generator with RAG") as app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>🎭 Mood-Based Story Generator</h1>")
        gr.Markdown("<p style='text-align: center;'>An immersive, multimodal storytelling experience powered by LLaMA-3, Stable Diffusion, and RAG.</p>")

        history_state = gr.State([])
        lore_state = gr.State()

        with gr.Row():
            prompt_input = gr.Textbox(lines=3, label="Enter your story prompt", placeholder="e.g., A lone knight approaches the glowing cave...")
            
            with gr.Column():
                genre_input = gr.Dropdown(choices=genres, label="Select Genre", value="Fantasy")
                mood_input = gr.Textbox(label="Enter Mood", placeholder="e.g., dark, hopeful, mysterious")
        
        with gr.Accordion("📚 Lore Settings (RAG)", open=False):
            with gr.Row():
                lore_upload = gr.File(label="Upload Lore (.pdf or .txt)", file_types=[".pdf", ".txt"])
                rag_enabled = gr.Checkbox(label="Enable RAG Integration", value=True)

        with gr.Row():
            gen_btn = gr.Button("✨ Generate / Continue Story", variant="primary")
            clear_btn = gr.Button("🗑️ Clear Story")

        with gr.Row():
            audio_btn = gr.Button("🎧 Narrate Latest Story")
            download_btn = gr.Button("⬇️ Download Story + Images Zip")

        with gr.Tabs():
            with gr.TabItem("📖 Story Chat"):
                chat_output = gr.Chatbot(label="Story History")
            with gr.TabItem("🖼️ Image Gallery"):
                gallery_output = gr.Gallery(label="Story Illustrations", columns=2)
            
        audio_output = gr.Audio(label="🔊 Voice Narration", type="filepath")
        download_output = gr.File(label="📦 Zip Bundle")

        # Event Listeners
        lore_upload.change(fn=process_lore, inputs=lore_upload, outputs=lore_state)
        gen_btn.click(
            fn=generate_story_with_history,
            inputs=[prompt_input, genre_input, mood_input, history_state, lore_state, rag_enabled],
            outputs=[history_state, chat_output, gallery_output]
        )
        clear_btn.click(fn=clear_history, inputs=[], outputs=[history_state, chat_output, gallery_output])
        audio_btn.click(fn=generate_narration, inputs=[history_state], outputs=[audio_output])
        download_btn.click(fn=download_zip, inputs=[history_state], outputs=[download_output])

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()
