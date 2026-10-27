# Хоррор-рилсы на автомате

Конвейер делает готовый вертикальный рилс (1080×1920, 40–60 с):

1. **История**: Claude придумывает короткую страшилку от первого лица с крючком в первой строке, раскладкой по сценам и подписью с хештегами.
2. **Озвучка**: офлайн-голос Piper (русский, `ru-irinia-medium`), немного понижен и с эхом.
3. **Звук**: синтезированный гул, ветер, резкие стингеры на страшных моментах, сердцебиение к финалу.
4. **Видео**: фоны рисуются кодом (телефон, комната, дверь, коридор, зеркало, лес, окно), силуэт с глазами, медленный зум, туман, зерно, мерцание, глитч, крупные субтитры.

## Запуск

```bash
sudo apt install ffmpeg fonts-dejavu-core
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # ключ с console.anthropic.com
python make_reel.py                      # новая история
python make_reel.py --theme "лифт ночью"
python make_reel.py --story stories/sample.json   # готовая история без ИИ
```

Результат: `out/<дата>/reel.mp4`, `cover.jpg`, `caption.txt`, `story.json`. Голосовая модель (60 МБ) скачивается сама при первом запуске.

Модель для историй можно сменить переменной `REELS_MODEL`.

## Каждый день автоматически (GitHub Actions)

Workflow `.github/workflows/daily-reel.yml` собирает новый рилс каждый день в 18:00 по Москве.

1. Добавь ключ: Settings → Secrets and variables → Actions → New repository secret, имя `ANTHROPIC_API_KEY`.
2. Готовый рилс появится во вкладке Actions → последний запуск → Artifacts (`reel-N`, хранится 14 дней).
3. Запустить вручную, с темой или без: Actions → Daily horror reel → Run workflow.
