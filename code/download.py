import requests
import time
from pathlib import Path
from tqdm import tqdm

START_ID = 1105348
END_ID = 1106348

SAVE_DIR = Path("habr_articles")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html",
    "Accept-Language": "ru-RU,ru;q=0.9",
})

for article_id in tqdm(range(START_ID, END_ID + 1)):
    url = f"https://habr.com/ru/articles/{article_id}/"
    file_path = SAVE_DIR / f"{article_id}.html"

    if file_path.exists():
        continue

    try:
        response = session.get(url, timeout=20)

        if response.status_code == 404:
            continue

        if response.status_code != 200:
            tqdm.write(f"[FAIL] {article_id}: {response.status_code}")
            continue

        if "text/html" not in response.headers.get("Content-Type", ""):
            continue

        file_path.write_text(response.text, encoding="utf-8")

    except requests.RequestException as e:
        tqdm.write(f"[ERROR] {article_id}: {e}")

    time.sleep(0.5)

print("Готово!")
print(f"Скачано HTML-файлов: {len(list(SAVE_DIR.glob('*.html')))}")