# Data layout

Each experiment is isolated by tag:

```
datasets/<tag>/
├── manifest.json
├── heldout.jsonl
└── <dataset>/train.jsonl
```

Rows use the canonical schema:

```json
{"sample_id": "...", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "noise_type": "none"}
```

`noise_type` is `none` for clean samples. Existing `dolly-ratio5` and `dolly-ratio10` data can be consumed directly by `python cli.py`.
