# Story evaluation: prompt v1 vs v2

12 premises x 2 chapters per version, same seed per premise, Llama-3.2-3B-Instruct running locally. Scores come from a zero-shot classifier (facebook/bart-large-mnli) and simple text statistics: they are proxies for comparing versions, not absolute quality. Each version has 24 chapters, so one chapter moves a percentage by ~4 points: treat small gaps as noise.

| Metric | v1 (before) | v2 (current) |
|---|---|---|
| Event probability (higher = more happens) | 0.97 | 0.98 |
| Mood correct, top-1 of 10 | 25% | 21% |
| Mood correct, top-3 of 10 | 46% | 42% |
| Genre correct, top-1 of 10 | 12% | 21% |
| Genre correct, top-3 of 10 | 50% | 67% |
| Ch. 2 phrases copied from ch. 1 (lower = better) | 1.3% | 0.1% |
| Repeated trigrams within a chapter (lower = better) | 0.5% | 0.4% |
| Chapters with dialogue | 88% | 100% |
| Chapters ending on a full sentence | 100% | 100% |
| Average words (target 280) | 269 | 252 |
