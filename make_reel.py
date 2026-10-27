"""Make one horror reel: story -> voice-over -> 9:16 video.

  python make_reel.py                       # Claude invents a new story (needs ANTHROPIC_API_KEY)
  python make_reel.py --theme "лифт ночью"  # same, with a theme
  python make_reel.py --story stories/sample.json
"""
import argparse
import json
import random
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from reels.audio import build_audio
from reels.config import ROOT, VOICE_MODEL, VOICE_URL
from reels.render import render_video
from reels.story import generate_story, load_story


def ensure_voice() -> None:
    if VOICE_MODEL.exists():
        return
    print("Downloading Russian voice model...")
    VOICE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "voice.tar.gz"
        urllib.request.urlretrieve(VOICE_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(VOICE_MODEL.parent, filter="data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", type=Path, help="Use a ready story JSON instead of generating one")
    ap.add_argument("--theme", help="Theme hint for the generated story")
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    ensure_voice()
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    story = load_story(args.story) if args.story else generate_story(args.theme)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.out / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "story.json").write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "caption.txt").write_text(story.get("caption", ""), encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[{story['title']}] voicing {len(story['lines'])} lines...")
        audio, timeline, total = build_audio(story, Path(tmp), seed)
        print(f"rendering {total:.1f}s video...")
        render_video(story, timeline, total, audio, outdir / "reel.mp4", seed)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "3", "-i", str(outdir / "reel.mp4"),
                    "-frames:v", "1", str(outdir / "cover.jpg")], check=True)
    print(f"done: {outdir / 'reel.mp4'}")


if __name__ == "__main__":
    main()
