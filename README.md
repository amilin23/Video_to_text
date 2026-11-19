# Video to Text

Watches a webcam or a video file and uses a Qwen vision model from a hosted
provider to **list the objects it sees** and **describe what is happening**:
what people are doing and what happens to the objects. Results are shown on the
video, printed in the terminal and saved to files.

You can also **ask questions about the video by voice** and hear the answer.

```
video ──► a few frames every few seconds ──► Qwen (your provider) ──► objects + description
```

## What you need

- **Windows** with **Python 3.10 or newer**
- From your Qwen provider: an **API key**, the **address** (base URL, ending in `/v1`)
  and the **model name** of a vision model
- For voice questions (optional): an **ElevenLabs API key**, a microphone, and **FFmpeg**

## Setup (once)

Run these in PowerShell.

**1. Go to the project folder**

```powershell
cd "C:\path\to\Video_to_text"
```

**2. Create and activate a virtual environment**

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. Activate it again every time you open a new PowerShell window.

**3. Install the packages**

```powershell
pip install -r requirements.txt
```

**4. Add your keys**

```powershell
copy .env.example .env
```

Open `.env` in a text editor and fill in your provider's details:

```
QWEN_API_KEY=your-provider-key
QWEN_BASE_URL=https://your-provider-address/v1
QWEN_MODEL=your-model-name
ELEVENLABS_API_KEY=your-elevenlabs-key   # only for voice questions
```

`.env` is never uploaded to git, so your keys stay on your computer.

**5. For voice questions only: install FFmpeg**

```powershell
winget install Gyan.FFmpeg
```

## Run it

**1. Put your videos** (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) in the **`videos`** folder.

**2. Start the app**

```powershell
python live_camera.py
```

The first time, this also creates your `settings.py`.

**3. Pick a video** from the list, or `0` for the webcam:

```
Choose what to analyse:
  0) Webcam
  1) desk.mp4      0:26
  2) walking.mp4   0:25
Number (Enter = 1, q = quit):
```

**4. Use the keys** in the video window (click the window first):

| Key | What it does |
|---|---|
| **Space** | Analyse now |
| **A** | Turn automatic analysis on or off |
| **V** | Ask a question by voice: tap V, speak, tap V again |
| **Q** | Quit |

## Two modes

Set in `settings.py` with `FULL`:

| | Live mode (`FULL = False`) | Full mode (`FULL = True`) |
|---|---|---|
| How it works | Plays the video in real time and describes it every few seconds | Describes every 5-second piece of the video, one after another |
| Covers | Only part of the video (the moments it analyses) | The whole video |
| Speed | Real time | As fast as the model answers |
| Extras | Voice questions, YOLO boxes | A short summary of the whole video at the end |
| Best for | The webcam, demos | Recorded videos |

## Results

Everything is saved in the **`results`** folder, named after the video and the time:

| File | Contains |
|---|---|
| `<video>_<time>.jsonl` | Every description with its time in the video |
| `<video>_<time>_summary.txt` | Summary of the whole video (full mode) |
| `<video>_<time>_questions.txt` | Your voice questions and the answers |
| `<video>_<time>.srt` | Subtitles to play with the video, if `SRT = 'auto'` |

## Settings

All settings are in **`settings.py`**, with a comment explaining each one. Change a
value, save, and run the app again. The most useful ones:

| Setting | What it does |
|---|---|
| `SOURCE` | `'choose'` = show the list, `0` = webcam, or a file name like `'walking.mp4'` |
| `FULL` | `False` = live mode, `True` = full mode |
| `SEGMENT` | Full mode: seconds per piece (default 5) |
| `INTERVAL` | Live mode: seconds between analyses (default 3) |
| `FRAMES` | Frames sent per analysis (default 4) |
| `MAX_SIDE` | Size of the frames sent, in pixels (default 640); larger reads small text better but is slower |
| `SYSTEM_PROMPT` | The instructions given to Qwen |
| `EXTRA_BODY` | Provider-specific options, e.g. turning Qwen's thinking off for faster answers |
| `MAX_TOKENS` | Longest answer allowed; raise it if answers come back empty |
| `SKIP_UNCHANGED` | Don't ask again when the picture hasn't changed (saves requests) |
| `YOLO` | `'yolov8n.pt'` = also draw object boxes (live mode, needs `pip install ultralytics`) |
| `VOICE_ID` | The ElevenLabs voice that speaks the answers |
| `LOG`, `SRT`, `SUMMARY` | Which result files to save |

Text goes in quotes, yes/no is `True` / `False`, and Windows paths use forward
slashes (`'C:/videos/clip.mp4'`). A mistake gives a clear message, for example
`unknown setting FRAMS. Did you mean FRAMES?`.

`settings.py` is your own file: `git pull` never changes it. After an update, new
settings appear in `settings.example.py`; copy any you want into your `settings.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No API key` / `401` | Check `QWEN_API_KEY` in `.env` |
| `Model '...' not found` | Check `QWEN_MODEL` in `.env`; the message lists the available models |
| `Server returned 400` after changing `EXTRA_BODY` | Your provider doesn't accept that option; remove it or set `EXTRA_BODY = None` |
| Answers are empty | Raise `MAX_TOKENS`, or turn thinking off in `EXTRA_BODY` |
| Answers are slow | Turn thinking off, lower `FRAMES` or `MAX_SIDE` |
| A video isn't in the list | Check it is in the `videos` folder and ends in `.mp4`, `.mov`, `.avi`, `.mkv` or `.webm` |
| Keys don't work | Click the video window first, then press the key |
| Camera won't open | Close other apps using the camera, or set `SOURCE = 1` |
| `git pull` says local changes would be overwritten | Commit your changes first: `git add -A`, then `git commit -m "my changes"`, then `git pull` |

## Privacy

Frames from your video go to your Qwen provider, and voice questions go to
ElevenLabs. Use videos you are allowed to share, and ask permission before
filming other people.
