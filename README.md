# Video to Text

Watches a webcam or a video file and uses a Qwen vision model from a hosted
provider to **list the objects in view** and **describe what is happening**:
what people are doing and what happens to the objects. Results are shown on the
video, printed in the terminal and saved to files.

Questions about the video can also be **asked by voice**, with the answer spoken aloud.

```
video ──► a few frames every few seconds ──► Qwen (hosted provider) ──► objects + description
```

## Requirements

- **Windows** with **Python 3.10 or newer**
- From a Qwen provider: an **API key**, the **base URL** (ending in `/v1`)
  and the **name of a vision model**
- For voice questions (optional): an **ElevenLabs API key**, a microphone and **FFmpeg**

## Setup

Run the following commands in PowerShell.

**1. Get the project**

```powershell
git clone https://github.com/amilin23/Video_to_text.git
```

```powershell
cd Video_to_text
```

Alternatively, download the ZIP from GitHub (**Code → Download ZIP**) and open the extracted folder.

**2. Create and activate a virtual environment**

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

The prompt then starts with `(.venv)`. Activate the environment again in every new PowerShell window.

**3. Install the packages**

```powershell
pip install -r requirements.txt
```

**4. Add the API keys**

```powershell
copy .env.example .env
```

Open `.env` in a text editor and fill in the provider details:

```
QWEN_API_KEY=provider-api-key
QWEN_BASE_URL=https://provider-address/v1
QWEN_MODEL=model-name

# Optional: only needed for voice questions
ELEVENLABS_API_KEY=elevenlabs-api-key
```

`.env` is excluded from git, so the keys are never uploaded.
Without an ElevenLabs key, voice questions are switched off and everything else works normally.

**5. Install FFmpeg (only for voice questions)**

```powershell
winget install Gyan.FFmpeg
```

## Usage

**1. Add videos** (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) to the **`videos`** folder.
A few sample videos (`sample_*.mp4`) are included.

**2. Start the app**

```powershell
python live_camera.py
```

On the first run, this also creates `settings.py`.

**3. Choose a video** from the list, or `0` for the webcam:

```
Choose what to analyse:
  0) Webcam
  1) sample_action.mp4    0:26
  2) sample_busy.mp4      0:25
Number (Enter = 1, q = quit):
```

**4. Controls** (click the video window first):

| Key | Action |
|---|---|
| **Space** | Analyse now |
| **A** | Turn automatic analysis on or off |
| **V** | Ask a question by voice: tap V, speak, tap V again |
| **Q** | Quit |

## Modes

Selected in `settings.py` with `FULL`:

| | Live mode (`FULL = False`) | Full mode (`FULL = True`) |
|---|---|---|
| How it works | Plays the video in real time and describes it every few seconds | Describes every 5-second section of the video in order |
| Coverage | Only the moments that are analysed | The whole video |
| Speed | Real time | As fast as the model responds |
| Extras | Voice questions, YOLO boxes | Summary of the whole video at the end |
| Best for | Webcam, demos | Recorded videos |

## Output

All results are saved in the **`results`** folder, named after the video and the time of the run:

| File | Contents |
|---|---|
| `<video>_<time>.jsonl` | Every description with its time in the video |
| `<video>_<time>_summary.txt` | Summary of the whole video (full mode) |
| `<video>_<time>_questions.txt` | Voice questions and their answers |
| `<video>_<time>.srt` | Subtitles to play alongside the video (when `SRT = 'auto'`) |

## Configuration

All settings are in **`settings.py`**, each with a comment explaining it. Change a
value, save the file and run the app again. The most useful settings:

| Setting | Description |
|---|---|
| `SOURCE` | `'choose'` = show the list, `0` = webcam, or a file name such as `'sample_busy.mp4'` |
| `FULL` | `False` = live mode, `True` = full mode |
| `SEGMENT` | Full mode: length of each section in seconds (default 5) |
| `INTERVAL` | Live mode: seconds between analyses (default 3) |
| `FRAMES` | Frames sent per analysis (default 4) |
| `MAX_SIDE` | Size of the frames sent, in pixels (default 640); larger reads small text better but is slower |
| `SYSTEM_PROMPT` | Instructions given to the model |
| `EXTRA_BODY` | Provider-specific options, e.g. turning off Qwen's thinking for faster answers |
| `MAX_TOKENS` | Maximum answer length; increase it if answers come back empty |
| `SKIP_UNCHANGED` | Skip analysis when the picture has not changed (saves requests) |
| `YOLO` | `'yolov8n.pt'` = also draw object boxes (live mode, requires `pip install ultralytics`) |
| `VOICE_ID` | ElevenLabs voice used for spoken answers |
| `LOG`, `SRT`, `SUMMARY` | Which result files are saved |

Text values go in quotes, yes/no values are `True` / `False`, and Windows paths use
forward slashes (`'C:/videos/clip.mp4'`). Mistakes produce a clear message, for
example `unknown setting FRAMS. Did you mean FRAMES?`.

`settings.py` is local and not tracked by git, so `git pull` never changes it. New
settings added in an update appear in `settings.example.py` and can be copied into `settings.py`.

## Troubleshooting

| Problem | Solution |
|---|---|
| `No API key` / `401` | Check `QWEN_API_KEY` in `.env` |
| `Model '...' not found` | Check `QWEN_MODEL` in `.env`; the message lists the available models |
| `Server returned 400` after changing `EXTRA_BODY` | The provider does not accept that option; remove it or set `EXTRA_BODY = None` |
| Empty answers | Increase `MAX_TOKENS`, or turn off thinking in `EXTRA_BODY` |
| Slow answers | Turn off thinking, or lower `FRAMES` or `MAX_SIDE` |
| A video is missing from the list | Check that it is in the `videos` folder and ends in `.mp4`, `.mov`, `.avi`, `.mkv` or `.webm` |
| Keys do not respond | Click the video window first, then press the key |
| Camera does not open | Close other apps using the camera, or set `SOURCE = 1` |
| `git pull` reports that local changes would be overwritten | Commit local changes first (`git add -A`, then `git commit -m "local changes"`), then run `git pull` |

## Privacy

Frames from the video are sent to the Qwen provider, and voice questions are sent
to ElevenLabs. Only use videos that may be shared, and ask for permission before
filming other people.