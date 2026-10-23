"""Write the example story shown by the app's "Read an example" button.

It runs the app's own pipeline: the sample world's lore, the default text and image models, and
the first suggested choice for every chapter after the first. The text is kept exactly as written;
the committed pictures were picked from a few seeds at the app's own settings (SD-Turbo, one step),
since a one-step model's results vary a lot from seed to seed. Run from the project root:
    python -m scripts.make_showcase
"""
import json
import shutil

from app import main

CHAPTERS = 3
ART_STYLE = "Storybook painting"


def final_state(steps):
    *_, last = steps
    return last[0]


def main_():
    lore, _, _, premise, genre, mood = main.load_sample_world()
    story = main.new_story()
    for n in range(CHAPTERS):
        prompt = premise if n == 0 else story["choices"][0]
        story = final_state(main.write_chapter(prompt, genre, mood, ART_STYLE, story, lore, True, "Medium"))
        if len(story["chapters"]) != n + 1:
            raise SystemExit(f"chapter {n + 1} failed; see the warning above")
        print(f"chapter {n + 1}: {len(story['chapters'][-1]['text'].split())} words, "
              f"lore used: {story['chapters'][-1]['lore_used']}")

    out = main.SHOWCASE_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for n, chapter in enumerate(story["chapters"], 1):
        if chapter["image"]:
            name = f"chapter_{n}.jpg"
            shutil.copy(chapter["image"], out / name)
            chapter["image"] = name
    (out / "story.json").write_text(json.dumps(story, indent=2, ensure_ascii=False) + "\n")
    print(f"saved to {out}")


if __name__ == "__main__":
    main_()
