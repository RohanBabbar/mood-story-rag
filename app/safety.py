import re

# A deliberately small blocklist for clearly unacceptable requests. The models also get
# instructions to keep things general-audience; this just stops the obvious cases early.
_MINORS = r"(child|children|kid|kids|minor|minors|underage|loli|shota|preteen)"
_SEXUAL = r"(sex|sexual|nude|nudity|porn\w*|explicit|erotic\w*|nsfw|hentai)"
_BLOCKED = [
    re.compile(rf"\b{_MINORS}\b.*\b{_SEXUAL}\b|\b{_SEXUAL}\b.*\b{_MINORS}\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(porn\w*|nsfw|hentai|sex scene|explicit sex|erotica)\b", re.IGNORECASE),
]


def check_prompt(text):
    """Return a user-facing reason if the prompt should be refused, else None."""
    if any(pattern.search(text or "") for pattern in _BLOCKED):
        return "This app keeps stories suitable for a general audience, so it can't write that. Try a different premise."
    return None
