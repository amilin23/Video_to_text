import cv2
import numpy as np

from live_camera import Results, fmt_time, read_segments, srt_time
from qwen_client import Analysis


def make_video(path, seconds, fps=10, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(int(seconds * fps)):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_fmt_time():
    assert fmt_time(0) == "0:00"
    assert fmt_time(65.9) == "1:05"
    assert fmt_time(3723) == "1:02:03"


def test_srt_time():
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(3723.456) == "01:02:03,456"


def test_segments_cover_whole_video_without_gaps(tmp_path):
    video = tmp_path / "clip.avi"
    make_video(video, seconds=12, fps=10)
    cap = cv2.VideoCapture(str(video))
    segments = list(read_segments(cap, segment_s=5, n_frames=4))
    cap.release()

    spans = [(start, end) for start, end, _ in segments]
    assert spans[0] == (0, 5)
    assert spans[1] == (5, 10)
    # Last segment is cut at the end of the video (12 s), not padded to 15 s.
    assert spans[2][0] == 10 and abs(spans[2][1] - 12) < 0.01
    assert [len(f) for _, _, f in segments] == [4, 4, 2]


def test_last_segment_never_ends_after_the_video(tmp_path):
    # All 3 sample points of the last segment (10-15 s) fall before the end (14.5 s).
    video = tmp_path / "clip.avi"
    make_video(video, seconds=14.5, fps=10)
    cap = cv2.VideoCapture(str(video))
    segments = list(read_segments(cap, segment_s=5, n_frames=3))
    cap.release()
    assert len(segments) == 3
    assert abs(segments[-1][1] - 14.5) < 0.01


def test_srt_file(tmp_path):
    srt = tmp_path / "out.srt"
    results = Results("m", None, str(srt))
    results.add(Analysis(["cup"], "A cup on a desk.", 1.0, ""), 0, 5, srt_end=5)
    results.add(Analysis([], "Someone picks it up.", 1.0, ""), 5, 10, srt_end=10)
    results.write_srt()
    assert srt.read_text() == (
        "1\n00:00:00,000 --> 00:00:05,000\nObjects: cup\nA cup on a desk.\n"
        "\n"
        "2\n00:00:05,000 --> 00:00:10,000\nSomeone picks it up.\n"
    )


def test_live_mode_subtitles_last_until_next_entry(tmp_path):
    srt = tmp_path / "out.srt"
    results = Results("m", None, str(srt))
    results.add(Analysis([], "First.", 1.0, ""), 1.0, 2.5)
    results.add(Analysis([], "Second.", 1.0, ""), 8.0, 9.5)
    results.write_srt()
    text = srt.read_text()
    assert "00:00:01,000 --> 00:00:08,000" in text
    assert "00:00:08,000 --> 00:00:13,000" in text


def test_scene_change_detects_real_changes_only():
    from live_camera import scene_change, scene_signature
    rng = np.random.default_rng(0)
    still = [np.full((360, 640, 3), 120, dtype=np.uint8)] * 4
    noisy = [np.clip(f.astype(int) + rng.integers(-3, 4, f.shape), 0, 255).astype(np.uint8) for f in still]
    moved = [np.full((360, 640, 3), 120, dtype=np.uint8) for _ in range(4)]
    for f in moved:
        f[100:300, 200:500] = 255  # a bright object appears
    base = scene_signature(still)
    assert scene_change(scene_signature(noisy), base) < 6      # camera noise: unchanged
    assert scene_change(scene_signature(moved), base) > 6      # new object: changed
    assert scene_change(scene_signature(moved[:2]), base) > 6  # different frame counts still work
