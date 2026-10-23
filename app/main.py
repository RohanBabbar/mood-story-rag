import html
import random
import threading
import time
import uuid
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from app.config import (
    ADAPTER_REPO,
    BACKEND,
    CHAPTERS_PER_DAY,
    FILE_RETENTION_HOURS,
    IMAGE_MODEL,
    IMAGE_PROVIDER,
    MAX_LORE_FILE_MB,
    MAX_LORE_FILES,
    MAX_LORE_WORDS,
    MAX_PROMPT_CHARS,
    OUTPUT_DIR,
    REQUIRE_LOGIN,
    TEXT_MODEL,
)
from app.llm import (
    LENGTHS,
    StoryError,
    analyze_chapter,
    build_messages,
    explain_hf_error,
    stream_story,
    trim_to_last_sentence,
)
from app.multimodal import (
    ART_STYLES,
    DEFAULT_ART_STYLE,
    NO_ILLUSTRATIONS,
    first_sentences,
    generate_image,
    generate_narration,
    load_image_pipeline,
)
from app.rag_pipeline import build_faiss_index, get_embedder, retrieve_context
from app.safety import check_prompt
from app.utils import SUPPORTED_LORE_TYPES, chunk_text, delete_old_files, load_custom_lore

GENRES = ["Fantasy", "Sci-Fi", "Mystery", "Adventure", "Romance", "Horror", "Comedy", "Drama", "Fairy Tale", "Noir"]
MOODS = ["Hopeful", "Dark", "Whimsical", "Eerie", "Melancholic", "Tense", "Cozy", "Epic", "Bittersweet", "Playful"]

IDEAS = [
    ("A lighthouse keeper finds a door at the bottom of the sea that wasn't there yesterday.", "Fantasy", "Eerie"),
    ("The last botanist on a generation ship discovers the plants have started to whisper.", "Sci-Fi", "Melancholic"),
    ("A retired detective's cat keeps bringing home clues to a murder nobody has reported.", "Mystery", "Whimsical"),
    ("Two rival bakers are snowed in together the night before the village festival.", "Romance", "Cozy"),
    ("A girl and her cat run a tiny bakery on an island that floats above the clouds.", "Fantasy", "Whimsical"),
    ("Every night at 3:14 a.m., the radio in the abandoned station plays tomorrow's news.", "Horror", "Tense"),
    ("A disgraced cartographer is hired to map a city that rearranges itself every sunrise.", "Adventure", "Epic"),
    ("An old robot who repairs umbrellas falls in love with the rain.", "Sci-Fi", "Bittersweet"),
]

PROMPT_PLACEHOLDER = "A lone knight approaches a cave that hums with forgotten music..."
NEXT_PLACEHOLDER = "What happens next? Leave empty to let the story decide, or pick a suggestion."

EMPTY_STORY = """
<div class="empty">
  <p class="empty-title">Nothing written yet.</p>
  <p>Give it a premise (or grab a random idea), choose a mood, and press <b>Start the story</b>.
  Each chapter gets its own illustration, and you decide where it goes next.</p>
</div>
"""

NO_LORE_STATUS = "No lore loaded. The story can invent its own world."

FINE_TUNED, BASE_MODEL = "Fine-tuned (LoRA)", "Base model"


def new_story():
    return {"chapters": [], "cast": {}, "summary": "", "choices": []}


# === Rendering ===

def _roman(n):
    numerals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in numerals:
        while n >= value:
            out += symbol
            n -= value
    return out


def render_story(chapters, phase=None):
    """Render chapters as HTML. `phase` marks the last chapter as 'writing' or 'illustrating'."""
    if not chapters:
        return EMPTY_STORY
    parts = []
    for i, chapter in enumerate(chapters, 1):
        is_last = i == len(chapters)
        if chapter["image"]:
            figure = f'<img class="art" src="/gradio_api/file={quote(chapter["image"])}" alt="Illustration for chapter {i}">'
        elif is_last and phase == "illustrating":
            figure = '<div class="art art-pending">sketching the scene...</div>'
        else:
            figure = ""

        paragraphs = [p.strip() for p in chapter["text"].split("\n") if p.strip()]
        body = "".join(f"<p>{html.escape(p)}</p>" for p in paragraphs)
        if is_last and phase == "writing":
            body = (body or "<p></p>")[:-4] + '<span class="cursor"></span></p>'

        meta = [chapter["genre"], chapter["mood"], chapter["style"] if chapter["image"] else "",
                chapter.get("model", "")]
        meta = " · ".join(html.escape(m.lower()) for m in meta if m)
        lore = '<span class="lore-mark" title="Grounded in your uploaded lore">lore</span>' if chapter["lore_used"] else ""
        direction = f'<p class="direction">{html.escape(chapter["prompt"])}</p>' if chapter["prompt"] else ""

        parts.append(
            f'<section class="chapter"><div class="chapter-head"><span class="numeral">{_roman(i)}.</span>'
            f'<span class="meta">{meta}{lore}</span></div>{direction}{figure}<div class="chapter-text">{body}</div></section>'
        )
    return '<div class="book">' + '<div class="fleuron">&#8258;</div>'.join(parts) + "</div>"


def render_notes(story):
    """The running summary and cast list shown under the story."""
    if not story["chapters"] or not (story["summary"] or story["cast"]):
        return ""
    out = '<div class="notes">'
    if story["summary"]:
        out += f'<div class="note"><h4>The story so far</h4><p>{html.escape(story["summary"])}</p></div>'
    if story["cast"]:
        people = "".join(
            f"<li><b>{html.escape(name)}</b> {html.escape(look)}</li>" for name, look in story["cast"].items()
        )
        out += f'<div class="note"><h4>Cast</h4><ul>{people}</ul></div>'
    return out + "</div>"


def story_view(story, phase=None, prompt=None):
    """All UI outputs for a story state. While `phase` is set, the story is still being written."""
    busy = phase is not None
    choices = [] if busy else story["choices"]
    if busy:
        button = gr.update(value="Writing...", interactive=False)
    else:
        button = gr.update(value="Write next chapter" if story["chapters"] else "Start the story", interactive=True)
    choice_buttons = [gr.update(value=c, visible=True) for c in choices]
    choice_buttons += [gr.update(visible=False)] * (3 - len(choices))
    return (
        story,
        render_story(story["chapters"], phase),
        "" if busy else render_notes(story),
        prompt if prompt is not None else gr.update(),
        button,
        gr.update(visible=bool(choices)),
        *choice_buttons,
        gr.update(visible=bool(story["chapters"]) and not busy),
    )


# === Callbacks ===

def process_lore(files):
    """Index uploaded lore files for retrieval. Returns (lore_index, status_markdown)."""
    if not files:
        return None, NO_LORE_STATUS
    chunks, sources, problems = [], [], []
    if len(files) > MAX_LORE_FILES:
        problems.append(f"only the first {MAX_LORE_FILES} files are used")
        files = files[:MAX_LORE_FILES]
    words_left = MAX_LORE_WORDS
    for path in files:
        name = Path(path).name
        if Path(path).stat().st_size > MAX_LORE_FILE_MB * 1024 * 1024:
            problems.append(f"{name}: larger than {MAX_LORE_FILE_MB} MB")
            continue
        if words_left <= 0:
            problems.append(f"{name}: lore word limit reached")
            continue
        try:
            words = load_custom_lore(path).split()
            if len(words) > words_left:
                problems.append(f"{name}: only the first {words_left:,} words are used")
                words = words[:words_left]
            words_left -= len(words)
            file_chunks = chunk_text(" ".join(words))
        except ValueError as e:
            problems.append(str(e))
            continue
        except Exception as e:
            problems.append(f"{name}: could not be read ({e})")
            continue
        chunks.extend(file_chunks)
        sources.append(name)

    if not chunks:
        return None, "Couldn't use that: " + " · ".join(problems)
    lore = build_faiss_index(chunks, sources)
    status = (f"Loaded {len(sources)} file{'s' if len(sources) > 1 else ''}, "
              f"{len(chunks)} passage{'s' if len(chunks) > 1 else ''}. Chapters will stick to it.")
    if problems:
        status += "\n\nSkipped: " + " · ".join(problems)
    return lore, status


class DailyLimiter:
    """Caps chapters per visitor per day. In memory, so it resets when the app restarts."""

    def __init__(self, limit):
        self.limit = limit
        self.counts = defaultdict(int)
        self.day = date.today()
        self.lock = threading.Lock()

    def take(self, key):
        with self.lock:
            if date.today() != self.day:
                self.counts.clear()
                self.day = date.today()
            if self.counts[key] >= self.limit:
                return False
            self.counts[key] += 1
            return True


limiter = DailyLimiter(CHAPTERS_PER_DAY)


def visitor_id(request):
    if request is None:
        return "local"
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def inference_token(oauth_token, request):
    """Pick whose HF token pays for this chapter, enforcing login or the daily cap."""
    if REQUIRE_LOGIN:
        if oauth_token is None:
            raise StoryError("Sign in with Hugging Face first (button at the top). "
                             "Stories use your own free Hugging Face inference credits.")
        return oauth_token.token
    if CHAPTERS_PER_DAY and not limiter.take(visitor_id(request)):
        raise StoryError(f"That's the daily limit of {CHAPTERS_PER_DAY} chapters. Come back tomorrow!")
    return None  # use the server's own HF_TOKEN


def notice(message, story):
    """Show a problem without failing the event.

    Raising gr.Error would stamp an "Error" badge on every output component, which makes the whole
    page look broken for what is usually an expected situation (not signed in, a refused prompt...).
    """
    gr.Warning(message)
    yield story_view(story)


def write_chapter(prompt, genre, mood, art_style, story, lore, use_lore, length, model_choice=FINE_TUNED,
                  request: gr.Request = None, oauth_token: gr.OAuthToken | None = None):
    """Stream the next chapter (or the first one), then analyse and illustrate it."""
    prompt = (prompt or "").strip()
    mood = (mood or "").strip()
    original = story or new_story()
    try:
        if not original["chapters"] and not prompt:
            raise StoryError("Give the story a premise first, or try a random idea.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise StoryError(f"Please keep the premise under {MAX_PROMPT_CHARS} characters.")
        refusal = check_prompt(" ".join([prompt, genre or "", mood]))
        if refusal:
            raise StoryError(refusal)
        token = inference_token(oauth_token, request)
    except StoryError as e:
        yield from notice(str(e), original)
        return

    # Work on a copy so a failed generation leaves the existing story untouched.
    story = {**original, "chapters": list(original["chapters"]), "cast": dict(original["cast"]), "choices": []}
    previous = [c["text"] for c in story["chapters"]]

    context = ""
    if use_lore and lore is not None:
        query = prompt or previous[-1][-600:]
        context = retrieve_context(query, lore)

    messages = build_messages(prompt, genre, mood, context, previous, length, story["cast"], story["summary"])
    chapter = {"prompt": prompt, "genre": genre, "mood": mood, "style": art_style, "text": "", "image": None,
               "lore_used": bool(context),
               # Which model wrote it, so base and fine-tuned chapters can be compared side by side.
               "model": ("fine-tuned" if model_choice == FINE_TUNED else "base") if ADAPTER_REPO else ""}
    story["chapters"].append(chapter)

    last_render = 0.0
    try:
        for piece in stream_story(messages, length, token, use_adapter=model_choice == FINE_TUNED):
            chapter["text"] += piece
            # Throttle UI updates; re-rendering on every token is wasteful.
            if time.monotonic() - last_render > 0.08:
                last_render = time.monotonic()
                yield story_view(story, "writing")
        chapter["text"] = trim_to_last_sentence(chapter["text"])
        if not chapter["text"]:
            raise StoryError("The model returned an empty chapter. Please try again.")
    except Exception as e:
        yield from notice(str(e) if isinstance(e, StoryError) else f"Story generation failed: {e}", original)
        return

    illustrate = art_style != NO_ILLUSTRATIONS
    yield story_view(story, "illustrating" if illustrate else "thinking")

    try:
        notes = analyze_chapter(chapter["text"], story["summary"], story["cast"], token)
        story.update(summary=notes["summary"], cast=notes["cast"], choices=notes["choices"])
    except Exception as e:
        print(f"[analysis failed] {e}")
        notes = {"scene": "", "scene_characters": []}

    if illustrate:
        scene = notes["scene"] or first_sentences(chapter["text"])
        characters = [(n, story["cast"][n]) for n in notes["scene_characters"] if n in story["cast"]][:3]
        try:
            chapter["image"] = generate_image(scene, genre, mood or "evocative", art_style, characters, token)
        except Exception as e:
            reason = explain_hf_error(e) if hasattr(e, "response") else str(e)
            gr.Warning(f"Couldn't illustrate this chapter: {reason}")

    yield story_view(story, prompt=gr.update(value="", placeholder=NEXT_PLACEHOLDER))


def follow_choice(choice, genre, mood, art_style, story, lore, use_lore, length, model_choice=FINE_TUNED,
                  request: gr.Request = None, oauth_token: gr.OAuthToken | None = None):
    yield from write_chapter(choice, genre, mood, art_style, story, lore, use_lore, length, model_choice,
                             request, oauth_token)


def random_idea():
    premise, genre, mood = random.choice(IDEAS)
    return premise, genre, mood


def narrate_latest(story):
    if not story["chapters"]:
        raise gr.Error("There's no story to narrate yet.")
    try:
        return gr.update(value=generate_narration(story["chapters"][-1]["text"]), visible=True)
    except Exception as e:
        print(f"[narration failed] {e}")
        raise gr.Error("The narration service didn't respond. Give it a minute and try again.") from e


def export_story(story):
    """Bundle the story as Markdown plus illustrations in a zip."""
    if not story["chapters"]:
        raise gr.Error("There's no story to export yet.")
    # The random suffix keeps two visitors exporting in the same second from sharing a file.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    zip_path = OUTPUT_DIR / f"mood-story-{stamp}-{uuid.uuid4().hex[:8]}.zip"
    lines = ["# My Mood Story", ""]
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i, chapter in enumerate(story["chapters"], 1):
            lines += [f"## Chapter {_roman(i)}", f"*{chapter['genre']} · {chapter['mood'] or 'any mood'}*", ""]
            if chapter["image"] and Path(chapter["image"]).exists():
                image_name = f"chapter_{i}.jpg"
                zf.write(chapter["image"], f"images/{image_name}")
                lines += [f"![Chapter {i}](images/{image_name})", ""]
            lines += [chapter["text"], ""]
        if story["summary"]:
            lines += ["---", "", "## The story so far", "", story["summary"], ""]
        if story["cast"]:
            lines += ["## Cast", ""] + [f"- **{n}**: {look}" for n, look in story["cast"].items()] + [""]
        zf.writestr("story.md", "\n".join(lines))
    return gr.update(value=str(zip_path), visible=True)


def reset_story():
    return (
        *story_view(new_story(), prompt=gr.update(value="", placeholder=PROMPT_PLACEHOLDER)),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
    )


# === UI ===

INK, PAPER, ACCENT = "#2a2520", "#f4efe6", "#b4532a"
INK_DARK, PAPER_DARK, ACCENT_DARK = "#e9e2d6", "#1c1a17", "#de7b4f"

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill=PAPER, body_background_fill_dark=PAPER_DARK,
    body_text_color=INK, body_text_color_dark=INK_DARK,
    body_text_color_subdued="#7a6f63", body_text_color_subdued_dark="#9b9184",
    background_fill_primary=PAPER, background_fill_primary_dark=PAPER_DARK,
    background_fill_secondary="#ece5d8", background_fill_secondary_dark="#26231f",
    border_color_primary="#e0d6c5", border_color_primary_dark="#36312b",
    block_background_fill="transparent", block_background_fill_dark="transparent",
    block_border_width="0px", block_shadow="none", block_shadow_dark="none",
    block_padding="0px",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
    block_title_padding="0 0 6px 0", block_label_padding="0 0 6px 0",
    block_title_text_color="#7a6f63", block_title_text_color_dark="#9b9184",
    input_background_fill="#fbf8f2", input_background_fill_dark="#24211d",
    input_border_color="#dbd0be", input_border_color_dark="#3a352f",
    input_border_color_focus=ACCENT, input_border_color_focus_dark=ACCENT_DARK,
    color_accent=ACCENT, color_accent_soft="#f0dccd", color_accent_soft_dark="#3a2a22",
    button_primary_background_fill=INK, button_primary_background_fill_dark=INK_DARK,
    button_primary_background_fill_hover=ACCENT, button_primary_background_fill_hover_dark=ACCENT_DARK,
    button_primary_text_color=PAPER, button_primary_text_color_dark=PAPER_DARK,
    button_primary_border_color="transparent", button_primary_border_color_dark="transparent",
    button_secondary_background_fill="transparent", button_secondary_background_fill_dark="transparent",
    button_secondary_background_fill_hover="#ece5d8", button_secondary_background_fill_hover_dark="#26231f",
    button_secondary_border_color="#cdbfa9", button_secondary_border_color_dark="#3f3933",
    button_secondary_text_color=INK, button_secondary_text_color_dark=INK_DARK,
    checkbox_background_color_selected=ACCENT, checkbox_background_color_selected_dark=ACCENT_DARK,
    checkbox_border_color_selected=ACCENT, checkbox_border_color_selected_dark=ACCENT_DARK,
    link_text_color=ACCENT, link_text_color_dark=ACCENT_DARK,
)

HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;'
    '1,6..72,400&display=swap" rel="stylesheet">'
)

CSS = (Path(__file__).parent / "style.css").read_text()


def create_app():
    credits = f"{TEXT_MODEL.split('/')[-1]} + {IMAGE_MODEL.split('/')[-1]}, MiniLM + FAISS for lore"
    # The Llama license asks apps built on it to say so.
    llama_notice = " · Built with Llama" if "llama" in TEXT_MODEL.lower() else ""

    # Uploaded lore files live in Gradio's cache; clear them out on the same schedule as our own files.
    retention = int(FILE_RETENTION_HOURS * 3600)
    with gr.Blocks(title="mood story", delete_cache=(3600, retention)) as demo:
        story_state = gr.State(new_story())
        lore_state = gr.State(None)

        gr.HTML(
            '<header class="masthead"><span class="wordmark">mood story</span>'
            '<span class="tagline">illustrated stories that remember their own world</span></header>'
        )
        if REQUIRE_LOGIN:
            with gr.Row(elem_classes="signin"):
                gr.HTML('<p class="hint">Sign in to write. Stories run on your own free Hugging Face credits.</p>')
                gr.LoginButton(size="sm")

        with gr.Row(equal_height=False, elem_classes="layout"):
            with gr.Column(scale=4, min_width=300, elem_classes="sidebar"):
                prompt_input = gr.Textbox(lines=4, label="Premise", placeholder=PROMPT_PLACEHOLDER,
                                          max_length=MAX_PROMPT_CHARS)
                with gr.Row():
                    genre_input = gr.Dropdown(GENRES, value="Fantasy", label="Genre", allow_custom_value=True, min_width=120)
                    mood_input = gr.Dropdown(MOODS, value="Hopeful", label="Mood", allow_custom_value=True, min_width=120)
                with gr.Row():
                    art_style_input = gr.Dropdown(list(ART_STYLES), value=DEFAULT_ART_STYLE, label="Art style", min_width=120)
                    length_input = gr.Dropdown(list(LENGTHS), value="Medium", label="Chapter length", min_width=120)
                model_input = gr.Radio([FINE_TUNED, BASE_MODEL], value=FINE_TUNED, label="Story model",
                                       info="Your LoRA adapter, or the same model without it",
                                       visible=bool(ADAPTER_REPO))

                write_btn = gr.Button("Start the story", variant="primary")
                random_btn = gr.Button("Random idea", size="sm", elem_classes="quiet")

                with gr.Accordion("World lore", open=False, elem_classes="lore"):
                    gr.Markdown("Upload notes about your world (PDF, TXT or MD). The most relevant bits are "
                                "fed to every chapter so names, places and rules stay consistent.",
                                elem_classes="hint")
                    lore_upload = gr.File(show_label=False, file_types=SUPPORTED_LORE_TYPES,
                                          file_count="multiple", height=110)
                    lore_status = gr.Markdown(NO_LORE_STATUS, elem_classes="hint")
                    use_lore = gr.Checkbox(value=True, label="Use lore when writing")

            with gr.Column(scale=7, min_width=340, elem_classes="page"):
                story_html = gr.HTML(EMPTY_STORY)
                with gr.Column(visible=False, elem_classes="choices") as choices_box:
                    gr.HTML('<p class="choices-label">What happens next?</p>')
                    choice_buttons = [
                        gr.Button(visible=False, size="sm", elem_classes="choice") for _ in range(3)
                    ]
                notes_html = gr.HTML("")
                with gr.Row(elem_classes="actions", visible=False) as actions_row:
                    narrate_btn = gr.Button("Read it aloud", size="sm", elem_classes="quiet")
                    export_btn = gr.Button("Download (.zip)", size="sm", elem_classes="quiet")
                    clear_btn = gr.Button("New story", size="sm", elem_classes="quiet")
                audio_output = gr.Audio(label="Narration", type="filepath", visible=False, buttons=["download"])
                download_output = gr.File(label="Your story", visible=False)

        gr.HTML(
            f'<footer class="colophon"><p>made by Rohan Babbar · {html.escape(credits)}{llama_notice}</p>'
            "<p>Stories and pictures are AI-generated and can be odd or wrong. Please keep it general-audience.</p>"
            "</footer>"
        )

        settings = [genre_input, mood_input, art_style_input, story_state, lore_state, use_lore, length_input,
                    model_input]
        outputs = [story_state, story_html, notes_html, prompt_input, write_btn, choices_box, *choice_buttons, actions_row]
        # A local GPU can only run one generation at a time; hosted APIs can handle several.
        concurrency = 1 if BACKEND == "local" else 4

        write_btn.click(write_chapter, inputs=[prompt_input, *settings], outputs=outputs,
                        concurrency_limit=concurrency, concurrency_id="write")
        for button in choice_buttons:
            button.click(follow_choice, inputs=[button, *settings], outputs=outputs,
                         concurrency_limit=concurrency, concurrency_id="write")
        random_btn.click(random_idea, outputs=[prompt_input, genre_input, mood_input])
        lore_upload.change(process_lore, inputs=lore_upload, outputs=[lore_state, lore_status])
        narrate_btn.click(narrate_latest, inputs=story_state, outputs=audio_output)
        export_btn.click(export_story, inputs=story_state, outputs=download_output)
        clear_btn.click(reset_story, outputs=[*outputs, audio_output, download_output])

    return demo


def _housekeeping():
    while True:
        delete_old_files(OUTPUT_DIR, FILE_RETENTION_HOURS)
        time.sleep(3600)


def launch():
    threading.Thread(target=_housekeeping, daemon=True).start()
    # Load the embedding model now rather than on the first lore upload, and the image model rather
    # than on the first picture (a ~5 GB download on a fresh Space).
    threading.Thread(target=get_embedder, daemon=True).start()
    if IMAGE_PROVIDER != "hf":
        threading.Thread(target=load_image_pipeline, daemon=True).start()
    create_app().queue().launch(theme=THEME, css=CSS, head=HEAD, allowed_paths=[str(OUTPUT_DIR)],
                                footer_links=[], max_file_size=f"{MAX_LORE_FILE_MB}mb")


if __name__ == "__main__":
    launch()
