# Football Score Tracker

Real-time football score tracker using **YOLO** and (future) **LLM** integration.

## Flow

```
MP4 video → extract frames every n sec/frame → YOLO + OCR detect overlay text → store in txt
```

Output format (delimiter `=============` with time/frame):

```
=============
frame=0
time=0.00s

MAN UNITED 2 - 1 LIVERPOOL
45' HALF TIME

=============
frame=30
time=1.00s

MAN UNITED 2 - 1 LIVERPOOL
46' SECOND HALF
```

## Future

- Replace txt storage with **LLM** calls to extract/interpret scores.
- `llm_client` module is prepared for this; implement `send_to_llm()` when ready.

## Project Structure

```
score_tracker/
├── video_processor.py   # MP4 → frame extraction
├── text_detector.py     # YOLO region detection + EasyOCR
├── storage.py           # Write to txt with delimiter
├── llm_client.py        # Stub for future LLM integration
├── pipeline.py          # Orchestrates the flow
main.py                  # CLI entrypoint
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Every 30 frames (~1 sec at 30fps)
python main.py match.mp4

# Every 5 seconds
python main.py match.mp4 --interval-sec 5.0

# Custom output and time range
python main.py match.mp4 -o scores.txt --start 0 --end 300
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `<video>_detections.txt` | Output txt path |
| `--interval-sec` | - | Extract 1 frame every N seconds |
| `--interval-frames` | 30 | Extract 1 frame every N frames |
| `--start` | 0 | Start time (sec) |
| `--end` | full | End time (sec) |
| `--yolo-model` | keremberke/yolov8n-table-extraction | YOLO model for region detection |
| `--no-yolo` | - | Full-frame OCR (includes stadium ads) |
| `--overlay-zones-only` | - | Only top/bottom 25% of frame (reduces ads) |
