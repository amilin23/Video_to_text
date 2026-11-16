# Settings for live_camera.py
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

# 0 = default webcam, 1 = second camera, or a video file, e.g. 'C:/videos/clip.mp4'
SOURCE = 0

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
SYSTEM_PROMPT = (
    "You are a live camera assistant. You receive consecutive frames from a camera, "
    "oldest first. Identify the main objects that are visible and explain what is "
    "happening, including any movement or change between frames. Be concise and "
    "only describe what you can actually see."
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

# Stop after this many seconds. 0 = run until you quit
MAX_RUNTIME = 0


# ------------------------------------------------------------ full mode (FULL = True)

# Length of each analysed piece of video, in seconds
SEGMENT = 5


# ------------------------------------------------------------ output

# Show the video window. False = just print the results
DISPLAY = True

# The window is shrunk to fit this width so the text stays on screen. 0 = original size
WINDOW_WIDTH = 1280

# Save every result to this file, one JSON line each. None = off, e.g. 'results.jsonl'
LOG = None

# Save descriptions as a subtitle file (video files only). None = off, e.g. 'video.srt'
SRT = None
