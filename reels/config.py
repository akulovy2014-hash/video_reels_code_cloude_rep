from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICE_MODEL = ROOT / "voices" / "ru-irinia-medium.onnx"
VOICE_URL = "https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ru-irinia-medium.tar.gz"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

W, H = 1080, 1920
FPS = 30
SR = 44100

# Pause before the first line and between lines, seconds
LEAD_IN = 0.25
LINE_GAP = 0.25
TAIL = 1.6

SCENES = ["phone", "room", "door", "corridor", "mirror", "forest", "window"]
