"""Voice-over (Piper TTS), synthesized ambient drone, stingers and the final mix."""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from .config import LEAD_IN, LINE_GAP, SR, TAIL, VOICE_MODEL

# Pitch the voice down a little and add a short room echo: sounds colder, still clear
VOICE_FILTER = "asetrate=22050*0.9,aresample=44100,atempo=1.1,aecho=0.8:0.6:45|90:0.22|0.12,volume=1.6"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        ch = w.getnchannels()
    data = data.astype(np.float32) / 32768.0
    return data.reshape(-1, ch).mean(axis=1) if ch > 1 else data


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def tts_line(text: str, out: Path) -> np.ndarray:
    raw = out.with_suffix(".raw.wav")
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", str(VOICE_MODEL), "-f", str(raw),
         "--length-scale", "1.05", "--noise-scale", "0.6"],
        input=text.encode("utf-8"), check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af", VOICE_FILTER,
         "-ac", "1", "-ar", str(SR), str(out)],
        check=True,
    )
    return read_wav(out)


def ambient(duration: float, rng: np.random.Generator) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    base = rng.choice([41.2, 43.65, 46.25, 49.0])
    drone = (
        0.5 * np.sin(2 * np.pi * base * t)
        + 0.35 * np.sin(2 * np.pi * (base * 1.012) * t)
        + 0.25 * np.sin(2 * np.pi * base * 1.5 * t + np.sin(2 * np.pi * 0.07 * t))
        + 0.12 * np.sin(2 * np.pi * base * 2.97 * t)
    )
    drone *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6))
    # Wind: smoothed noise with slow swells
    noise = rng.standard_normal(n // 32 + 2).astype(np.float32)
    wind = np.interp(np.arange(n) / 32, np.arange(len(noise)), noise)
    wind = np.convolve(wind, np.ones(64) / 64, mode="same")
    wind *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t + rng.uniform(0, 6)) ** 2
    # Faint eerie whistle
    whistle = 0.04 * np.sin(2 * np.pi * (880 + 6 * np.sin(2 * np.pi * 5 * t)) * t)
    whistle *= np.clip(np.sin(2 * np.pi * 0.03 * t), 0, 1) ** 3
    mix = 0.22 * drone + 0.9 * wind + whistle
    fade = np.minimum(1, np.minimum(t / 1.0, (duration - t) / 1.5))
    return (mix * fade).astype(np.float32)


def stinger(rng: np.random.Generator, length: float = 1.6) -> np.ndarray:
    n = int(length * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * 3.0)
    boom = np.sin(2 * np.pi * (60 * np.exp(-t * 1.5) + 30) * t) * np.exp(-t * 2.2)
    screech = rng.standard_normal(n) * np.exp(-t * 9)
    sweep = np.sin(2 * np.pi * (1400 * np.exp(-t * 2) + 200) * t) * env * 0.4
    s = 0.9 * boom + 0.35 * screech + sweep
    return (s / np.abs(s).max()).astype(np.float32)


def heartbeat(duration: float, bpm: float = 72) -> np.ndarray:
    n = int(duration * SR)
    out = np.zeros(n, dtype=np.float32)
    beat_len = int(0.18 * SR)
    tb = np.arange(beat_len) / SR
    thump = np.sin(2 * np.pi * 52 * tb) * np.exp(-tb * 28)
    period = 60 / bpm
    k = 0.0
    while k < duration:
        for offset, amp in ((0.0, 1.0), (0.24, 0.7)):
            i = int((k + offset) * SR)
            if i + beat_len < n:
                out[i:i + beat_len] += amp * thump
        k += period
    return out


def build_audio(story: dict, workdir: Path, seed: int) -> tuple[Path, list[tuple[float, float]], float]:
    """Returns (mixed wav path, [(start, end)] per line, total duration)."""
    rng = np.random.default_rng(seed)
    voices, timeline = [], []
    t = LEAD_IN
    for i, line in enumerate(story["lines"]):
        v = tts_line(line["text"], workdir / f"line{i:02d}.wav")
        # Trim near-silent edges so gaps are controlled by LINE_GAP
        idx = np.where(np.abs(v) > 0.01)[0]
        if len(idx):
            v = v[max(0, idx[0] - 800): idx[-1] + 2400]
        voices.append(v)
        dur = len(v) / SR
        timeline.append((t, t + dur))
        t += dur + LINE_GAP + (0.25 if line.get("scare") else 0)
    total = t - LINE_GAP + TAIL

    n = int(total * SR) + SR
    mix = ambient(n / SR, rng) * 0.35
    for v, (start, _) in zip(voices, timeline):
        i = int(start * SR)
        mix[i:i + len(v)] += v * 0.95

    def add(sound: np.ndarray, at: float, gain: float) -> None:
        i = max(0, int(at * SR))
        seg = sound[: max(0, n - i)]
        mix[i:i + len(seg)] += seg * gain

    add(stinger(rng), 0.0, 0.45)  # hook hit
    for line, (start, _) in zip(story["lines"], timeline):
        if line.get("scare"):
            add(stinger(rng), start - 0.05, 0.5)
    # Heartbeat over the last third builds tension into the ending
    hb_start = timeline[int(len(timeline) * 0.6)][0]
    hb = heartbeat(total - hb_start, bpm=78)
    hb *= np.linspace(0.15, 0.6, len(hb))
    add(hb, hb_start, 1.0)

    mix = mix[: int(total * SR)]
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    out = workdir / "mix.wav"
    write_wav(out, mix)
    return out, timeline, total
