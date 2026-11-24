# MiniMax H3 Workflow video model

Model: `minimax-h3-turbo`. Uses the existing asynchronous `/v1/videos` API.
Personal Hugging Face token required. First frame optional (text-only runtime
support must be verified); last frame not exposed. Gateway duration limit 1–10
seconds. Steps 4, 6 or 28. Seed -1 generates a random integer locally.
LoRA is disabled. The UI automatically polls every three seconds.

```bash
curl -X POST "$BASE_URL/v1/videos" \
  -H "X-HF-Token: $HF_TOKEN" \
  -F "model=minimax-h3-turbo" \
  -F "prompt=A forest in soft rain, birdsong" \
  -F "seconds=5" \
  -F "seed=-1" \
  -F "input_reference=@forest.png" \
  -F 'extra_body={"canvas":"960x544 · 16:9 fast","steps":6,"upsample":false}'
```

Use existing GET status/content routes. Audio is retained in the MP4 as
returned by the upstream. No live inference guarantee until verified.
