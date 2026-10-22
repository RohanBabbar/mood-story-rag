"""Story-quality evaluation: did the prompt redesign (v1 -> v2) make chapters better?

Each premise is written as a two-chapter story with both prompt versions, same seed, same local
model (Llama-3.2-3B). Chapters are then scored with free, local, reproducible measures:

- events:      zero-shot classifier (bart-large-mnli) probability that "something important happens"
- mood/genre:  zero-shot rank of the requested mood/genre among the app's options (top-1 / top-3)
- repetition:  share of chapter 2's 4-word phrases copied from chapter 1, and repeated trigrams
- dialogue, complete endings, length

Zero-shot scores are proxies, not human judgement: use them to compare versions, not as absolute
quality. ab_test.md holds blind pairs for a quick human check.

Run from the project root with MODEL_BACKEND=local:
    python -m eval.story_eval generate   # ~30-40 min on an M4; resumable
    python -m eval.story_eval score      # ~2 min
"""
import json
import random
import re
import sys
import time
from pathlib import Path

from app.llm import LENGTHS, MAX_CONTEXT_CHAPTERS, build_messages, stream_chat, trim_to_last_sentence
from app.main import GENRES, IDEAS, MOODS

RESULTS = Path(__file__).parent / "results"
SAMPLES = RESULTS / "story_samples.jsonl"
LENGTH = "Medium"

PREMISES = list(IDEAS) + [
    ("A man buys a strange banana at a market in a floating city.", "Fantasy", "Hopeful"),
    ("A night-shift nurse notices one patient's chart is written in her own handwriting.", "Mystery", "Tense"),
    ("Two rival magicians must share a stage after their theatre burns down.", "Comedy", "Playful"),
    ("A lonely astronaut receives a voicemail from her childhood home.", "Drama", "Melancholic"),
]


# === Prompt v1: the prompt as it was before the redesign ===
# Reconstructed exactly from the file contents read during development; it was never committed
# because the redesign landed before the first commit.

def build_messages_v1(prompt, genre, mood, previous_chapters=(), length=LENGTH):
    words = LENGTHS[length]
    system = (
        "You are a gifted novelist writing an episodic, illustrated story.\n"
        f"Genre: {genre}. Mood: {mood or 'author’s choice'}.\n"
        "Let the mood shape every choice: word choice, pacing, imagery and what characters notice.\n"
        f"Write about {words} words of vivid prose in short paragraphs. Use concrete sensory detail and "
        "at least one line of dialogue when characters are present.\n"
        "Output only the story text: no title, no chapter heading, no notes or commentary. "
        "End on a moment that invites the story to continue.\n"
        "Keep it suitable for a general audience: no explicit sexual content and no graphic gore. "
        "Even in horror, imply violence rather than describe it in detail."
    )
    if previous_chapters:
        story_so_far = "\n\n".join(previous_chapters[-MAX_CONTEXT_CHAPTERS:])
        user = (f"The most recent chapters:\n<story>\n{story_so_far}\n</story>\n\n"
                "Write the next chapter, picking up exactly where it left off without recapping.")
    else:
        user = f"Begin a new story from this premise: {prompt}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def write_chapter(version, premise, genre, mood, previous):
    """Generate one chapter with the full settings of that prompt version."""
    words = LENGTHS[LENGTH]
    if version == "v1":
        messages = build_messages_v1(premise if not previous else "", genre, mood, previous)
        settings = {"max_tokens": int(words * 1.6) + 80, "repetition_penalty": 1.0}
    else:
        messages = build_messages(premise if not previous else "", genre, mood, previous_chapters=previous,
                                  length=LENGTH)
        settings = {"max_tokens": words * 2 + 120, "repetition_penalty": 1.1}
    text = "".join(stream_chat(messages, settings["max_tokens"], temperature=0.85,
                               repetition_penalty=settings["repetition_penalty"]))
    return trim_to_last_sentence(text) if version == "v2" else text.strip()


def generate():
    import torch

    RESULTS.mkdir(exist_ok=True)
    done = {(s["premise_id"], s["version"]) for s in map(json.loads, SAMPLES.read_text().splitlines())} \
        if SAMPLES.exists() else set()
    for i, (premise, genre, mood) in enumerate(PREMISES):
        for version in ("v1", "v2"):
            if (i, version) in done:
                continue
            torch.manual_seed(1000 + i)  # same seed for both versions of a premise
            start = time.perf_counter()
            ch1 = write_chapter(version, premise, genre, mood, [])
            ch2 = write_chapter(version, premise, genre, mood, [ch1])
            sample = {"premise_id": i, "version": version, "premise": premise, "genre": genre, "mood": mood,
                      "chapters": [ch1, ch2], "seconds": round(time.perf_counter() - start, 1)}
            with SAMPLES.open("a") as f:
                f.write(json.dumps(sample) + "\n")
            print(f"[{i + 1}/{len(PREMISES)}] {version} {sample['seconds']}s  {premise[:50]}", flush=True)


# === Scoring ===

def words_of(text):
    return re.findall(r"[a-z']+", text.lower())


def ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def score():
    from transformers import pipeline

    samples = [json.loads(line) for line in SAMPLES.read_text().splitlines()]
    classify = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def rank(text, labels, target, template):
        result = classify(text, labels, hypothesis_template=template)
        return result["labels"].index(target)

    rows = []
    for s in samples:
        for n, text in enumerate(s["chapters"], 1):
            events = classify(text, ["something important happens", "nothing much happens"],
                              hypothesis_template="In this story, {}.")
            row = {
                "premise_id": s["premise_id"], "version": s["version"], "chapter": n,
                "event_prob": dict(zip(events["labels"], events["scores"], strict=True))["something important happens"],
                "mood_rank": rank(text, MOODS, s["mood"], "The mood of this story is {}."),
                "genre_rank": rank(text, GENRES, s["genre"], "This is a {} story."),
                "dialogue": bool(re.search(r"[\"“”]", text)),
                "complete_ending": text.rstrip()[-1:] in ".!?\"”’'…",
                "words": len(text.split()),
                "repeated_trigrams": 1 - len(set(ngrams(words_of(text), 3))) / max(len(ngrams(words_of(text), 3)), 1),
            }
            if n == 2:
                ch1_grams = set(ngrams(words_of(s["chapters"][0]), 4))
                ch2_grams = ngrams(words_of(text), 4)
                row["copied_from_ch1"] = sum(g in ch1_grams for g in ch2_grams) / max(len(ch2_grams), 1)
            rows.append(row)
        print(f"scored premise {s['premise_id']} {s['version']}", flush=True)

    def mean(version, key, chapter=None):
        values = [r[key] for r in rows if r["version"] == version and (chapter is None or r["chapter"] == chapter)]
        return sum(values) / len(values)

    def share(version, test):
        mine = [r for r in rows if r["version"] == version]
        return sum(test(r) for r in mine) / len(mine)

    metrics = [
        ("Event probability (higher = more happens)", lambda v: f"{mean(v, 'event_prob'):.2f}"),
        ("Mood correct, top-1 of 10", lambda v: f"{share(v, lambda r: r['mood_rank'] == 0):.0%}"),
        ("Mood correct, top-3 of 10", lambda v: f"{share(v, lambda r: r['mood_rank'] < 3):.0%}"),
        ("Genre correct, top-1 of 10", lambda v: f"{share(v, lambda r: r['genre_rank'] == 0):.0%}"),
        ("Genre correct, top-3 of 10", lambda v: f"{share(v, lambda r: r['genre_rank'] < 3):.0%}"),
        ("Ch. 2 phrases copied from ch. 1 (lower = better)", lambda v: f"{mean(v, 'copied_from_ch1', 2):.1%}"),
        ("Repeated trigrams within a chapter (lower = better)", lambda v: f"{mean(v, 'repeated_trigrams'):.1%}"),
        ("Chapters with dialogue", lambda v: f"{mean(v, 'dialogue'):.0%}"),
        ("Chapters ending on a full sentence", lambda v: f"{mean(v, 'complete_ending'):.0%}"),
        ("Average words (target 280)", lambda v: f"{mean(v, 'words'):.0f}"),
    ]
    n_premises = len({r["premise_id"] for r in rows})
    lines = [(f"# Story evaluation: prompt v1 vs v2\n\n{n_premises} premises x 2 chapters per version, "
             "same seed per premise, Llama-3.2-3B-Instruct running locally. Scores come from a zero-shot "
             "classifier (facebook/bart-large-mnli) and simple text statistics: they are proxies for comparing "
             "versions, not absolute quality. Each version has 24 chapters, so one chapter moves a percentage by "
             "~4 points: treat small gaps as noise.\n"),
             "| Metric | v1 (before) | v2 (current) |", "|---|---|---|"]
    lines += [f"| {name} | {fmt('v1')} | {fmt('v2')} |" for name, fmt in metrics]
    (RESULTS / "story_eval.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "story_eval.json").write_text(json.dumps(rows, indent=2))
    write_ab_test(samples)
    print("\n".join(lines))


def write_ab_test(samples):
    """Blind pairs for a human check: which chapter 1 is better? The key is kept in a separate file."""
    rng = random.Random(0)
    by_premise = {}
    for s in samples:
        by_premise.setdefault(s["premise_id"], {})[s["version"]] = s
    lines = ["# Blind A/B test\n", ("For each premise, read both openings and note which you'd rather keep reading "
             "(A or B). Don't open `ab_key.json` until you're done.\n")]
    key = {}
    for pid, versions in sorted(by_premise.items()):
        if set(versions) != {"v1", "v2"}:
            continue
        order = ["v1", "v2"]
        rng.shuffle(order)
        key[pid] = {"A": order[0], "B": order[1]}
        s = versions["v1"]
        lines.append(f"## {pid + 1}. {s['premise']} ({s['genre']}, {s['mood']})\n")
        for label, version in zip("AB", order, strict=True):
            lines.append(f"**{label}:**\n\n{versions[version]['chapters'][0]}\n")
    (RESULTS / "ab_test.md").write_text("\n".join(lines))
    (RESULTS / "ab_key.json").write_text(json.dumps(key, indent=2))


if __name__ == "__main__":
    {"generate": generate, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "generate"]()
