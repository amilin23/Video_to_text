# Settings for live_camera.py
#
# This is the TEMPLATE. The app copies it to settings.py the first time it runs.
# Edit settings.py, not this file: settings.py is yours and git never changes it.
#
# Change a value, save, and run:  python live_camera.py
# A command-line option overrides a value here for one run, e.g.  python live_camera.py --frames 2
#
# Rules:
#   - Text goes in quotes:          MODEL = 'qwen-qwen3-8-27b'
#   - Yes/no is True or False:      FULL = True
#   - None means "use the default"  (for BASE_URL and MODEL: use the value from your .env)
#   - Windows paths: use forward slashes, e.g. 'C:/videos/clip.mp4'
#   - The API key does NOT go here. It stays in .env as QWEN_API_KEY.


# ------------------------------------------------------------ input

# 'choose' = show a numbered list of the videos in VIDEO_FOLDER (and the webcam) and pick one
# 0 = default webcam, 1 = second camera
# or one video: a file name in VIDEO_FOLDER, e.g. 'walking.mp4', or a full path, e.g. 'C:/videos/clip.mp4'
SOURCE = 'choose'

# Folder with your test videos (.mp4, .mov, .avi, .mkv, .webm). Relative to the project folder.
VIDEO_FOLDER = 'videos'

# True  = analyse every part of a video file, 5 seconds at a time (video files only)
# False = live mode: play in real time and analyse a recent burst every few seconds
FULL = False


# ------------------------------------------------------------ model / provider

# Address of the model server. None = use QWEN_BASE_URL from your .env
BASE_URL = None

# Model name as your server knows it. None = use QWEN_MODEL from your .env
MODEL = None

# Extra fields sent with every request.
# This turns Qwen's thinking off, so answers come back faster.
# If you get "Server returned 400", delete one of the two lines inside,
# or set EXTRA_BODY = None to send nothing extra.
EXTRA_BODY = {
    'enable_thinking': False,
    'chat_template_kwargs': {'enable_thinking': False},
}

# Longest answer allowed, in tokens. With thinking off, 300 is enough.
# If answers come back empty, raise it (thinking may still be on).
MAX_TOKENS = 800

# 0 = most consistent answers, higher (e.g. 0.7) = more varied wording
TEMPERATURE = 0.2

# Seconds to wait for an answer before giving up
TIMEOUT = 120

# Send the previous description so the model can say what changed
USE_PREVIOUS = True

# Instructions given to the model. The app still asks for the answer as JSON
# with "objects" and "description", so keep it about describing frames.
# The rules below keep answers short (good for CIDEr), stop claims about things that
# aren't there, and keep object names consistent.
SYSTEM_PROMPT = (
    "You describe short clips of video. You receive a few frames from the clip, oldest first. "
    "Rules: "
    "1. Describe only what you can see in these frames. Never say what is absent or not "
    "happening, such as 'no people' or 'nothing moves'. "
    "2. Objects are things you could point at or pick up, such as a mug, a laptop or a person. "
    "Do not list surfaces, walls, floors, shadows, light, the camera or the scene itself. "
    "3. Use simple, common, singular names, and keep using the same name for the same object. "
    "4. The description is ONE sentence of at most 25 words: what is happening, including "
    "clear camera movement. If you are unsure what something is, say what it looks like."
)


# ------------------------------------------------------------ frames sent

# Frames sent per analysis. Fewer = faster, more = better at seeing motion
FRAMES = 4

# Frames are shrunk so their longest side is at most this many pixels.
# Smaller = faster, larger = can read smaller text
MAX_SIDE = 640

# JPEG quality of the frames sent, 1 to 100
JPEG_QUALITY = 80


# ------------------------------------------------------------ live mode (FULL = False)

# Seconds to wait after each answer before the next analysis. 0 = only when you press space
INTERVAL = 3

# Seconds between the frames in each burst (4 frames x 0.5 s = about 1.5 s of video)
FRAME_GAP = 0.5

# Draw YOLO boxes and pass the labels to Qwen. None = off, 'yolov8n.pt' = on
# (needs: pip install ultralytics)
YOLO = None

# How many times per second YOLO runs. Its last boxes stay on screen in between.
# Lower = smoother video on a slow computer
YOLO_FPS = 5

# Only show YOLO guesses (and pass them to Qwen) when it is at least this sure, 0 to 1
YOLO_CONFIDENCE = 0.5

# Stop after this many seconds. 0 = run until you quit
MAX_RUNTIME = 0


# ------------------------------------------------------------ full mode (FULL = True)

# Length of each analysed piece of video, in seconds
SEGMENT = 5

# After the whole video is analysed, write a short summary of it (one more request).
# Saved next to the log as <log name>_summary.txt
SUMMARY = True


# ------------------------------------------------------------ output

# Show the video window. False = just print the results
DISPLAY = True

# The window is shrunk to fit this width so the text stays on screen. 0 = original size
WINDOW_WIDTH = 1280

# Save every result, one JSON line each. Use this file with evaluate.py.
# 'auto' = a new file per run: results/<video name>_<date-time>.jsonl
# None = off, or a file name, e.g. 'results.jsonl' (adds to the end of the file)
LOG = 'auto'

# Save descriptions as a subtitle file (video files only).
# 'auto' = results/<video name>_<date-time>.srt, None = off, or a file name
SRT = None


# ------------------------------------------------------------ AI judge (evaluate.py --judge)

# The judge checks each description against the video frames and scores it 0-2.
# Best: a DIFFERENT model than the one being judged (a model judging itself is too lenient).
# None = use the same server / model as above.
JUDGE_BASE_URL = None
JUDGE_MODEL = None

# Extra request fields for the judge. None = same as EXTRA_BODY
JUDGE_EXTRA_BODY = None

# Frames the judge looks at per description
JUDGE_FRAMES = 4

# A judge on another service with its own key: put JUDGE_API_KEY in .env
