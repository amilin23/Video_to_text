# Video to text: live camera + Qwen

A rough prototype that watches your camera and uses a Qwen vision-language model
to **list the objects it sees** and **explain what is happening**, shown live on
the video and printed to the terminal.

```
camera ──► keep the last few frames (e.g. 4 frames, 0.5 s apart)
              │
              ├──► window shows live video + latest description
              │
              └──► every few seconds, in the background:
                     frames ──► Qwen (OpenAI-compatible API) ──► {"objects": [...], "description": "..."}
```

Sending a short burst of frames, rather than a single one, lets the model
describe motion ("a person picks up a cup") and not just a still scene. Each
request also includes the previous description, so the model can say what changed.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Run a Qwen vision model

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint that accepts
images works. Two common options:

**Ollama (easiest, default):**

```bash
ollama pull qwen3-vl:8b      # or qwen3-vl:4b for smaller GPUs
```

**vLLM (e.g. for Qwen3.8-27B, needs a large GPU):**

```bash
vllm serve Qwen/Qwen3.8-27B --port 8000
```

If vLLM complains about the number of images per prompt, add
`--limit-mm-per-prompt '{"image": 4}'`.

## 3. Start the camera

```bash
python live_camera.py
```

With vLLM instead of Ollama:

```bash
python live_camera.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3.8-27B
```

Keys: **space** analyse now · **a** toggle automatic analysis · **q** quit.

## Useful options

| Option | Default | What it does |
|---|---|---|
| `--source` | `0` | Camera index, or a video file / stream URL |
| `--interval` | `3` | Seconds to wait after each answer before the next one (`0` = only on space) |
| `--frames` | `4` | Frames sent per analysis |
| `--frame-gap` | `0.5` | Seconds between those frames |
| `--max-side` | `640` | Frames are downscaled to this size before sending (smaller = faster) |
| `--yolo [weights]` | off | Also draw YOLO boxes live and pass its labels to Qwen as hints (`pip install ultralytics`, AGPL-3.0) |
| `--log out.jsonl` | off | Save every result as a JSON line |
| `--no-display` | off | No window, just print results |
| `--extra-body JSON` | none | Extra request fields, e.g. turn off thinking on vLLM: `'{"chat_template_kwargs": {"enable_thinking": false}}'` |

## API key and settings (`.env`)

Copy the template and fill it in:

```bash
cp .env.example .env
```

```
QWEN_API_KEY=your-key-here
QWEN_BASE_URL=https://your-provider/v1
QWEN_MODEL=your-model-name
```

The app loads `.env` automatically from the folder you run it in (or the
project folder). `.env` is in `.gitignore`, so your key is never committed.
Variables already set in your terminal, and command-line options, take priority
over the file. For a local Ollama / vLLM server you can leave `QWEN_API_KEY` empty.

## Tips

- **Slow answers?** Use a smaller model (`qwen3-vl:4b`), fewer `--frames`, or a smaller `--max-side`.
- **Empty answers?** Thinking models can use up all their tokens on reasoning. Raise
  `--max-tokens` or turn thinking off (see `--extra-body`).
- **Camera won't open?** Try `--source 1`, and close other apps that use the camera.

## Tests

```bash
pip install pytest
pytest
```
