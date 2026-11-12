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

## 3. Start the camera (or see *Video files* below)

```bash
python live_camera.py
```

With vLLM instead of Ollama:

```bash
python live_camera.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3.8-27B
```

Keys: **space** analyse now · **a** toggle automatic analysis · **q** quit.

## Video files

### Play and sample (like the camera)

```bash
python live_camera.py --source my_video.mp4
```

The video plays at normal speed and a short burst is analysed every few
seconds. Results show the **position in the video**, e.g. `[1:05-1:06]`. Parts
of the video between analyses are not looked at.

### Analyse the whole video (`--full`)

```bash
python live_camera.py --source my_video.mp4 --full --srt my_video.srt --log my_video.jsonl
```

The video is split into back-to-back segments (`--segment`, 5 s by default)
and every segment is analysed in order, so nothing is skipped. It runs as fast
as the model answers, not at playback speed: a 10-minute video is 120 segments,
so at ~7 s per answer it takes about 14 minutes.

```
[0:00-0:05] (6.3s) Objects: menu panel, PLAY, OPTIONS
            The main menu is shown and the highlight moves from QUIT to PLAY.
[0:05-0:10] (7.1s) Objects: ...
```

### Check accuracy with subtitles (`--srt`)

`--srt file.srt` saves every description as a subtitle at the right time. Open
the video in VLC (or most players) and add the subtitle file
(Subtitle > Add Subtitle File...), or give it the same name as the video. You
then see each description while the matching part of the video plays, which
makes it easy to spot mistakes.

## Useful options

| Option | Default | What it does |
|---|---|---|
| `--source` | `0` | Camera index, or a video file / stream URL |
| `--full` | off | Video files: analyse every segment of the whole video |
| `--segment` | `5` | `--full`: seconds per analysed segment |
| `--interval` | `3` | Live: seconds to wait after each answer before the next one (`0` = only on space) |
| `--frames` | `4` | Frames sent per analysis |
| `--frame-gap` | `0.5` | Live: seconds between those frames |
| `--max-side` | `640` | Frames are downscaled to this size before sending (smaller = faster) |
| `--yolo [weights]` | off | Also draw YOLO boxes live and pass its labels to Qwen as hints (`pip install ultralytics`, AGPL-3.0) |
| `--log out.jsonl` | off | Save every result as a JSON line (with video times for files) |
| `--srt out.srt` | off | Video files: save descriptions as subtitles |
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
