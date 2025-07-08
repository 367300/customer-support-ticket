import argparse
import os
import sys
import requests

DATASETS = {
    "original": {
        "url": "https://disk.yandex.ru/d/GbebgnMOzkUD_Q",
        "filename": "customer_support_tickets.csv"
    },
    # В будущем можно добавить:
    # "translated": {
    #     "url": "ССЫЛКА_НА_ПЕРЕВЕДЕННЫЙ_ДАТАСЕТ",
    #     "filename": "customer_support_tickets_translated.csv"
    # }
}

def download_file(url, dest_path):
    print(f"Скачивание из {url} ...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Файл сохранён: {dest_path}")
    else:
        print(f"Ошибка скачивания! Код ответа: {response.status_code}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Скрипт для скачивания датасетов")
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True,
                        help="Какой датасет скачать: original (или translated в будущем)")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    dest_path = os.path.join(raw_dir, dataset["filename"])

    if os.path.exists(dest_path):
        print(f"Файл уже существует: {dest_path}")
    else:
        download_file(dataset["url"], dest_path)

if __name__ == "__main__":
    main()