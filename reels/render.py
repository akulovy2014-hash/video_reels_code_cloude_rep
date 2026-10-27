"""Frame renderer: slow zoom, fog, grain, flicker, glitches and burned-in subtitles, piped into ffmpeg."""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FONT_BOLD, FPS, H, W
from .visuals import SH, SW, render_scene

SUB_Y = int(H * 0.56)  # Centre of the subtitle block, clear of the Reels UI at the bottom
CROSSFADE = 0.35


def _smooth_noise(h: int, w: int, cell: int, rng) -> np.ndarray:
    small = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((w + 2 * cell, h + 2 * cell), Image.BICUBIC)
    return np.asarray(img, np.float32)[cell:cell + h, cell:cell + w] / 255.0


def _chunks(text: str, max_chars: int = 16) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def _text_sprite(text: str, size: int, fill, wrap: int = 0) -> tuple[np.ndarray, np.ndarray]:
    font = ImageFont.truetype(FONT_BOLD, size)
    lines = [text]
    if wrap:
        lines = _chunks(text, wrap)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    stroke = max(4, size // 11)
    boxes = [probe.textbbox((0, 0), ln, font=font, stroke_width=stroke) for ln in lines]
    lh = int(size * 1.18)
    tw = max(b[2] - b[0] for b in boxes) + 20
    th = lh * len(lines) + 20
    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, (ln, b) in enumerate(zip(lines, boxes)):
        x = (tw - (b[2] - b[0])) // 2 - b[0]
        d.text((x, 10 + i * lh), ln, font=font, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))
    a = np.asarray(img, np.float32)
    return a[..., :3], a[..., 3:] / 255.0


def _blit(frame: np.ndarray, sprite, cx: int, cy: int, alpha: float) -> None:
    rgb, a = sprite
    h, w = a.shape[:2]
    x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
    x1, y1 = min(W, x0 + w), min(H, y0 + h)
    rgb, a = rgb[: y1 - y0, : x1 - x0], a[: y1 - y0, : x1 - x0] * alpha
    region = frame[y0:y1, x0:x1]
    region *= 1 - a
    region += rgb * a


def render_video(story: dict, timeline, total: float, audio: Path, out: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lines = story["lines"]

    scenes = []
    for i, line in enumerate(lines):
        plain = render_scene(line["scene"], seed + i, False)
        fig = render_scene(line["scene"], seed + i, True) if line.get("figure") else None
        scenes.append((plain, fig))

    # Subtitle sprites with per-chunk timing proportional to chunk length
    subs = []
    for i, (line, (start, end)) in enumerate(zip(lines, timeline)):
        if i == 0:
            subs.append((start, end + 0.3, _text_sprite(line["text"].upper(), 74, (235, 60, 50), wrap=15), True))
            continue
        parts = _chunks(line["text"])
        weights = np.array([len(p) + 3 for p in parts], float)
        bounds = start + np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * (end - start)
        for p, s, e in zip(parts, bounds[:-1], bounds[1:]):
            subs.append((s, e, _text_sprite(p, 92, (245, 245, 240)), False))
    outro = _text_sprite("Подпишись, чтобы не пропустить следующую историю", 46, (200, 200, 200), wrap=26)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vignette = np.clip(1.25 - 1.05 * np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2), 0.18, 1)[..., None]
    fog_a = _smooth_noise(H, W * 2, 180, rng)
    fog_b = _smooth_noise(H, W * 2, 90, rng)
    grain = [rng.normal(0, 9, (H, W, 1)).astype(np.float32) for _ in range(6)]
    fog_color = np.array([150, 160, 170], np.float32)

    def line_at(t: float) -> int:
        idx = 0
        for i, (s, _) in enumerate(timeline):
            if t >= s - CROSSFADE:
                idx = i
        return idx

    def bg(i: int, t: float) -> np.ndarray:
        start = timeline[i][0]
        end = timeline[i + 1][0] if i + 1 < len(timeline) else total
        p = np.clip((t - start + CROSSFADE) / (end - start + CROSSFADE), 0, 1)
        plain, fig = scenes[i]
        img = plain
        # Figure appears in the middle of the line and blinks out again
        if fig is not None and 0.35 < p < 0.8 and not (0.6 < p < 0.63):
            img = fig
        z = 1.0 + 0.08 * p
        cw, ch = W * 1.1 / z, H * 1.1 / z
        drift = (i % 2 * 2 - 1) * 0.02 * SW * p
        cx, cy = SW / 2 + drift, SH / 2
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        return np.asarray(img.resize((W, H), Image.BILINEAR, box=box), np.float32)

    n_frames = int(total * FPS)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(audio),
         "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    flicker_until = -1.0
    for f in range(n_frames):
        t = f / FPS
        i = line_at(t)
        frame = bg(i, t)
        if i > 0 and t < timeline[i][0]:  # crossfade from the previous line's scene
            k = (timeline[i][0] - t) / CROSSFADE
            frame = frame * (1 - k) + bg(i - 1, t) * k

        off = int(t * 22) % W
        fog = fog_a[:, off:off + W] * 0.6 + fog_b[:, (2 * off) % W:(2 * off) % W + W] * 0.4
        frame += (fog[..., None] * 0.10) * (fog_color - frame) * 0.9

        # Lamp flicker: rare random dips, frequent during scare lines
        scare = lines[i].get("scare") and timeline[i][0] - 0.05 <= t <= timeline[i][0] + 0.6
        if scare or rng.random() < 0.006:
            flicker_until = t + rng.uniform(0.05, 0.25)
        bright = rng.uniform(0.35, 0.7) if t < flicker_until else 1.0
        frame *= bright * vignette

        if scare:
            shift = int(rng.integers(8, 30))
            frame[..., 0] = np.roll(frame[..., 0], shift, axis=1)
            frame[..., 2] = np.roll(frame[..., 2], -shift, axis=1)
            if rng.random() < 0.4:
                frame *= np.array([1.6, 0.6, 0.6], np.float32)
        if t < 0.12:  # hook flash on the very first frames
            frame += (1 - t / 0.12) * 120

        frame += grain[f % len(grain)]

        for s, e, sprite, is_hook in subs:
            if s <= t < e:
                a = min(1.0, (t - s) * FPS / 3)
                y = int(H * 0.45) if is_hook else SUB_Y
                _blit(frame, sprite, W // 2, y, a)
        if t > timeline[-1][1] + 0.3:
            _blit(frame, outro, W // 2, int(H * 0.7), min(1.0, (t - timeline[-1][1] - 0.3) * 2))

        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
