import numpy as np

from qwen_client import encode_frame, parse_response


def test_parses_plain_json():
    objects, desc = parse_response('{"objects": ["person", "laptop"], "description": "A person types."}')
    assert objects == ["person", "laptop"]
    assert desc == "A person types."


def test_strips_thinking_and_code_fences():
    text = '<think>hmm, let me look</think>\n```json\n{"objects": ["cup"], "description": "A cup on a desk."}\n```'
    assert parse_response(text) == (["cup"], "A cup on a desk.")


def test_handles_closing_think_tag_only():
    text = 'reasoning here</think>{"objects": "dog", "description": "A dog runs."}'
    assert parse_response(text) == (["dog"], "A dog runs.")


def test_falls_back_to_raw_text():
    assert parse_response("A person is waving at the camera.") == ([], "A person is waving at the camera.")


def test_encode_frame_downscales_to_jpeg_data_url():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    url = encode_frame(frame, max_side=640)
    assert url.startswith("data:image/jpeg;base64,")
