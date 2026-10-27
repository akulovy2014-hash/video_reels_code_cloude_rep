"""Story generation: Claude writes a short horror story as structured JSON."""
import json
import os
from pathlib import Path

from .config import SCENES

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "caption": {"type": "string"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "scene": {"type": "string", "enum": SCENES},
                    "figure": {"type": "boolean"},
                    "scare": {"type": "boolean"},
                },
                "required": ["text", "scene", "figure", "scare"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "caption", "lines"],
    "additionalProperties": False,
}

PROMPT = """Придумай новую оригинальную короткую хоррор-историю для вертикального видео (Instagram Reels), которую будет читать диктор.

Требования:
- Русский язык, от первого лица, разговорный стиль, как будто человек рассказывает то, что случилось с ним.
- Общая длина 100–125 слов (это 40–50 секунд озвучки).
- Первая строка — крючок, который за 2 секунды заставляет досмотреть: странное правило, запрет или пугающий факт (например «Никогда не отвечай, если…»). Не больше 10 слов.
- Дальше 8–11 коротких строк, по 1–2 предложения в каждой. Нарастание напряжения, конкретные детали (звуки, время, предметы).
- Предпоследняя строка — поворот или жуткая развязка. Последняя строка — короткая фраза, которая оставляет мурашки и возвращает к крючку.
- Без крови и жестокости, страх через атмосферу и неизвестность.
- Числа пиши словами, без цифр и без сокращений.
- Тема: {theme}

Для каждой строки выбери фон из списка: phone (светящийся телефон в темноте), room (тёмная комната), door (дверь со светом в щели), corridor (длинный коридор), mirror (зеркало), forest (ночной лес), window (окно с луной).
figure=true — если в кадре должен мелькнуть силуэт (не чаще 2–3 раз). scare=true — для 1–2 самых страшных строк (там будет резкий звук и глитч).
caption — подпись к рилсу: одна интригующая фраза и 5–8 хештегов."""

THEMES = [
    "звонок или сообщение посреди ночи",
    "что-то не так с соседями",
    "старая квартира и странные правила от прежних хозяев",
    "ночная смена на пустом объекте",
    "деревня, дача или лес",
    "отражения и зеркала",
    "голос, который зовёт по имени",
    "лифт, подъезд или лестничная клетка",
    "детский монитор или камера наблюдения",
    "сон, который начинает сбываться",
]


def generate_story(theme: str | None = None) -> dict:
    import random

    import anthropic

    theme = theme or random.choice(THEMES)
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=os.environ.get("REELS_MODEL", "claude-opus-5-5"),
        max_tokens=16000,
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        messages=[{"role": "user", "content": PROMPT.format(theme=theme)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Model declined to write the story; try another theme")
    text = next(b.text for b in response.content if b.type == "text")
    story = json.loads(text)
    story["theme"] = theme
    return story


def load_story(path: Path) -> dict:
    story = json.loads(Path(path).read_text(encoding="utf-8"))
    for line in story["lines"]:
        line.setdefault("figure", False)
        line.setdefault("scare", False)
        if line.get("scene") not in SCENES:
            line["scene"] = "room"
    return story
