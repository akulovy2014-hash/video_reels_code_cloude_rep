"""Procedural dark backgrounds: every scene is drawn from code, so no stock footage is needed."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import H, W

# Backgrounds are drawn 10% larger than the frame to allow a slow zoom
SW, SH = int(W * 1.1), int(H * 1.1)


def _grad(h: int, w: int, top, bottom) -> np.ndarray:
    k = np.linspace(0, 1, h)[:, None, None]
    return (np.array(top) * (1 - k) + np.array(bottom) * k) * np.ones((1, w, 1))


def _glow(arr: np.ndarray, cx: float, cy: float, radius: float, color, strength=1.0, sx=1.0, sy=1.0):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / radius
    arr += np.exp(-d ** 2)[..., None] * np.array(color) * strength


def _to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _blur(img: Image.Image, r: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(r))


def _silhouette(img: Image.Image, cx: float, base_y: float, height: float, eyes=True) -> Image.Image:
    """Tall dark figure with soft edges and two faint eyes."""
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    head_r = height * 0.075
    head_cy = base_y - height + head_r
    d.ellipse([cx - head_r * 0.85, head_cy - head_r, cx + head_r * 0.85, head_cy + head_r], fill=255)
    neck_y = head_cy + head_r * 0.9
    sh = height * 0.13
    d.polygon([
        (cx - head_r * 0.4, neck_y), (cx + head_r * 0.4, neck_y),
        (cx + sh, neck_y + height * 0.07), (cx + sh * 0.95, base_y),
        (cx - sh * 0.95, base_y), (cx - sh, neck_y + height * 0.07),
    ], fill=255)
    mask = _blur(mask, max(2, height * 0.012))
    dark = Image.new("RGB", img.size, (3, 3, 5))
    out = Image.composite(dark, img, mask)
    if eyes:
        e = Image.new("L", img.size, 0)
        de = ImageDraw.Draw(e)
        er = max(2, head_r * 0.09)
        for dx in (-0.32, 0.32):
            ex, ey = cx + dx * head_r, head_cy - head_r * 0.05
            de.ellipse([ex - er * 1.6, ey - er, ex + er * 1.6, ey + er], fill=210)
        e = _blur(e, er * 0.6)
        out = Image.composite(Image.new("RGB", img.size, (225, 225, 210)), out, e)
    return out


def scene_phone(rng, figure):
    a = _grad(SH, SW, (6, 8, 12), (2, 2, 4))
    px, py = SW * 0.5, SH * 0.66
    _glow(a, px, py, SW * 0.35, (40, 70, 110), 1.0, sx=1.4, sy=0.6)
    img = _to_img(a)
    d = ImageDraw.Draw(img)
    # Bed / table edge
    d.polygon([(0, SH * 0.6), (SW, SH * 0.57), (SW, SH), (0, SH)], fill=(10, 12, 17))
    a = np.asarray(img).astype(np.float32)
    _glow(a, px, py, SW * 0.22, (60, 100, 150), 1.0, sx=1.5, sy=0.5)
    img = _to_img(a)
    if figure:
        img = _silhouette(img, SW * 0.78, SH * 0.6, SH * 0.5)
    d = ImageDraw.Draw(img)
    pw, ph = SW * 0.2, SW * 0.38
    d.rounded_rectangle([px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2], radius=22, fill=(18, 20, 26))
    d.rounded_rectangle([px - pw / 2 + 8, py - ph / 2 + 8, px + pw / 2 - 8, py + ph / 2 - 8], radius=16, fill=(150, 190, 230))
    d.ellipse([px - 26, py + ph * 0.25, px + 26, py + ph * 0.25 + 52], fill=(200, 60, 60))
    d.ellipse([px - 30, py - ph * 0.18, px + 30, py - ph * 0.18 + 60], fill=(60, 80, 110))
    return _blur(img, 1.2)


def scene_room(rng, figure):
    a = _grad(SH, SW, (12, 14, 20), (5, 6, 9))
    img = _to_img(a)
    d = ImageDraw.Draw(img)
    floor_y = SH * 0.68
    d.rectangle([0, floor_y, SW, SH], fill=(7, 8, 11))
    # Moonlight patch on the floor
    d.polygon([(SW * 0.15, floor_y + 60), (SW * 0.5, floor_y + 60), (SW * 0.7, SH * 0.95), (SW * 0.2, SH * 0.95)], fill=(32, 40, 56))
    d.line([(SW * 0.33, floor_y + 60), (SW * 0.45, SH * 0.95)], fill=(10, 12, 16), width=14)
    # Bed and chair silhouettes
    d.rectangle([SW * 0.55, floor_y - SH * 0.08, SW * 1.05, floor_y + SH * 0.06], fill=(4, 4, 6))
    d.rectangle([SW * 0.92, floor_y - SH * 0.2, SW * 1.0, floor_y], fill=(3, 3, 5))
    d.rectangle([SW * 0.08, floor_y - SH * 0.16, SW * 0.11, floor_y + 20], fill=(3, 3, 5))
    d.rectangle([SW * 0.08, floor_y - SH * 0.05, SW * 0.26, floor_y - SH * 0.035], fill=(3, 3, 5))
    if figure:
        img = _silhouette(img, SW * 0.4, floor_y + 30, SH * 0.48)
    return _blur(img, 2)


def scene_door(rng, figure):
    a = _grad(SH, SW, (10, 11, 15), (4, 4, 6))
    img = _to_img(a)
    d = ImageDraw.Draw(img)
    dx0, dx1, dy0, dy1 = SW * 0.28, SW * 0.72, SH * 0.25, SH * 0.72
    d.rectangle([dx0 - 24, dy0 - 24, dx1 + 24, dy1], fill=(16, 17, 21))
    d.rectangle([dx0, dy0, dx1, dy1], fill=(22, 21, 24))
    for y0 in (0.3, 0.5):
        d.rectangle([dx0 + 40, SH * y0, dx1 - 40, SH * (y0 + 0.16)], outline=(15, 15, 18), width=8)
    d.ellipse([dx1 - 70, SH * 0.49, dx1 - 40, SH * 0.49 + 30], fill=(60, 55, 45))
    d.rectangle([0, dy1, SW, SH], fill=(6, 6, 8))
    img = _blur(img, 1.5)
    a = np.asarray(img).astype(np.float32)
    # Warm light leaking under the door
    _glow(a, SW * 0.5, dy1 + 4, SW * 0.25, (200, 120, 50), 0.9, sx=1.6, sy=0.06)
    _glow(a, SW * 0.5, dy1 + 60, SW * 0.3, (90, 50, 20), 0.5, sx=1.2, sy=0.25)
    if figure:
        # Two feet-shadows interrupting the light
        for fx in (0.44, 0.56):
            a[int(dy1 - 6):int(dy1 + 14), int(SW * fx - 40):int(SW * fx + 40)] *= 0.15
    return _blur(_to_img(a), 1)


def scene_corridor(rng, figure):
    a = np.zeros((SH, SW, 3), np.float32)
    img = _to_img(a)
    d = ImageDraw.Draw(img)
    vx, vy = SW * 0.5, SH * 0.45
    ex, ey = SW * 0.09, SH * 0.07  # far end half-size
    far = [vx - ex, vy - ey, vx + ex, vy + ey]
    d.polygon([(0, 0), (far[0], far[1]), (far[0], far[3]), (0, SH)], fill=(14, 17, 18))
    d.polygon([(SW, 0), (far[2], far[1]), (far[2], far[3]), (SW, SH)], fill=(11, 14, 15))
    d.polygon([(0, 0), (SW, 0), (far[2], far[1]), (far[0], far[1])], fill=(8, 9, 10))
    d.polygon([(0, SH), (SW, SH), (far[2], far[3]), (far[0], far[3])], fill=(9, 10, 10))
    d.rectangle(far, fill=(40, 44, 40))
    # Doors along the walls
    for k in (0.25, 0.5, 0.7):
        x = far[0] * k
        yt, yb = far[1] * k + SH * 0.18 * (1 - k), SH - (SH - far[3]) * k - SH * 0.02
        d.polygon([(x, yt), (x + SW * 0.06 * (1 - k * 0.6), yt + 20), (x + SW * 0.06 * (1 - k * 0.6), yb - 10), (x, yb)], fill=(6, 7, 8))
    img = _blur(img, 2)
    a = np.asarray(img).astype(np.float32)
    _glow(a, vx, vy, SW * 0.16, (90, 100, 80), 0.8)
    for k in (0.35, 0.65, 0.85):
        _glow(a, vx, vy - (vy - SH * 0.02) * k, SW * 0.12 * k, (70, 80, 70), 0.5, sx=1.5, sy=0.4)
    img = _to_img(a)
    if figure:
        img = _silhouette(img, vx + SW * 0.01, far[3] + 4, (far[3] - far[1]) * 0.95, eyes=False)
    return _blur(img, 1)


def scene_mirror(rng, figure):
    a = _grad(SH, SW, (12, 15, 16), (6, 7, 8))
    img = _to_img(a)
    d = ImageDraw.Draw(img)
    for x in range(0, SW, 130):
        d.line([(x, 0), (x, SH)], fill=(8, 10, 11), width=4)
    for y in range(0, SH, 130):
        d.line([(0, y), (SW, y)], fill=(8, 10, 11), width=4)
    mx0, mx1, my0, my1 = SW * 0.22, SW * 0.78, SH * 0.16, SH * 0.62
    d.ellipse([mx0 - 20, my0 - 20, mx1 + 20, my1 + 20], fill=(20, 20, 22))
    d.ellipse([mx0, my0, mx1, my1], fill=(26, 32, 36))
    d.rectangle([SW * 0.1, SH * 0.7, SW * 0.9, SH * 0.74], fill=(30, 32, 34))
    d.rectangle([0, SH * 0.74, SW, SH], fill=(5, 5, 6))
    if figure:
        inner = img.crop((int(mx0), int(my0), int(mx1), int(my1)))
        inner = _silhouette(inner, inner.width * 0.5, inner.height * 1.05, inner.height * 0.85)
        m = Image.new("L", img.size, 0)
        ImageDraw.Draw(m).ellipse([mx0, my0, mx1, my1], fill=255)
        layer = img.copy()
        layer.paste(inner, (int(mx0), int(my0)))
        img = Image.composite(layer, img, m)
    a = np.asarray(_blur(img, 1.5)).astype(np.float32)
    _glow(a, SW * 0.4, SH * 0.3, SW * 0.25, (40, 50, 55), 0.6)
    return _to_img(a)


def scene_forest(rng, figure):
    a = _grad(SH, SW, (14, 20, 26), (4, 6, 7))
    _glow(a, SW * 0.7, SH * 0.18, SW * 0.08, (190, 200, 210), 1.0)
    _glow(a, SW * 0.7, SH * 0.18, SW * 0.4, (40, 50, 60), 0.7)
    img = _to_img(a)
    for layer, (shade, count, wmin, wmax) in enumerate([(26, 26, 10, 26), (16, 16, 24, 50), (6, 9, 50, 120)]):
        d = ImageDraw.Draw(img)
        for _ in range(count):
            x = rng.uniform(-50, SW + 50)
            w = rng.uniform(wmin, wmax)
            d.polygon([(x - w * 0.4, 0), (x + w * 0.4, 0), (x + w * 0.6, SH), (x - w * 0.6, SH)], fill=(shade, shade + 2, shade + 4))
            for _ in range(3):
                by = rng.uniform(0.05, 0.5) * SH
                side = rng.choice([-1, 1])
                d.line([(x, by), (x + side * w * rng.uniform(2, 4), by - rng.uniform(40, 120))], fill=(shade, shade + 2, shade + 4), width=max(2, int(w * 0.15)))
        if layer == 1 and figure:
            img = _silhouette(img, SW * 0.38, SH * 0.82, SH * 0.32)
        img = _blur(img, 2.5 - layer)
    a = np.asarray(img).astype(np.float32)
    k = np.linspace(0, 1, SH)[:, None, None] ** 3
    a += k * np.array([30, 36, 40])  # ground fog
    return _to_img(a)


def scene_window(rng, figure):
    a = _grad(SH, SW, (8, 9, 13), (4, 4, 6))
    img = _to_img(a)
    wx0, wx1, wy0, wy1 = SW * 0.2, SW * 0.8, SH * 0.18, SH * 0.6
    sky = _grad(int(wy1 - wy0), int(wx1 - wx0), (22, 30, 44), (10, 14, 22))
    _glow(sky, sky.shape[1] * 0.65, sky.shape[0] * 0.3, sky.shape[1] * 0.07, (210, 215, 220), 1.0)
    _glow(sky, sky.shape[1] * 0.65, sky.shape[0] * 0.3, sky.shape[1] * 0.35, (50, 60, 80), 0.6)
    sky_img = _to_img(sky)
    if figure:
        sky_img = _silhouette(sky_img, sky_img.width * 0.35, sky_img.height * 1.1, sky_img.height * 0.9)
    img.paste(sky_img, (int(wx0), int(wy0)))
    d = ImageDraw.Draw(img)
    d.rectangle([wx0, wy0, wx1, wy1], outline=(14, 14, 16), width=30)
    d.line([((wx0 + wx1) / 2, wy0), ((wx0 + wx1) / 2, wy1)], fill=(14, 14, 16), width=22)
    d.line([(wx0, (wy0 + wy1) / 2), (wx1, (wy0 + wy1) / 2)], fill=(14, 14, 16), width=22)
    d.polygon([(wx0 - 120, wy0 - 60), (wx0 + 60, wy0 - 60), (wx0 + 10, SH * 0.75), (wx0 - 160, SH * 0.78)], fill=(5, 5, 7))
    d.polygon([(wx1 - 60, wy0 - 60), (wx1 + 120, wy0 - 60), (wx1 + 160, SH * 0.78), (wx1 - 20, SH * 0.75)], fill=(5, 5, 7))
    d.rectangle([wx0 - 40, wy1, wx1 + 40, wy1 + 30], fill=(18, 18, 20))
    return _blur(img, 1.5)


SCENE_FUNCS = {
    "phone": scene_phone, "room": scene_room, "door": scene_door, "corridor": scene_corridor,
    "mirror": scene_mirror, "forest": scene_forest, "window": scene_window,
}


def render_scene(name: str, seed: int, figure: bool) -> Image.Image:
    return SCENE_FUNCS[name](np.random.default_rng(seed), figure)
