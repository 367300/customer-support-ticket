import argparse
import os
import sys
import requests

DATASETS = {
    "original": {
        "public_url": "https://disk.yandex.ru/d/GbebgnMOzkUD_Q",
        "filename": "customer_support_tickets.csv",
        "save_dir": "data/raw"
    },
    "translated": {
        "public_url": "https://disk.yandex.ru/d/9GNABDyrbm58Jg", 
        "filename": "customer_support_tickets_translated.csv",
        "save_dir": "data/raw"
    },
    "train_data": {
        "public_url": "https://disk.yandex.ru/d/z1ynMVMmxoamuw",
        "filename": "train_data.csv",
        "save_dir": "data/raw"
    },
    "test_data": {
        "public_url": "https://disk.yandex.ru/d/OzsSvYv1yD_IQg", 
        "filename": "test_data.csv",
        "save_dir": "data/raw"
    }
}

def get_yandex_disk_download_link(public_url: str):
    """
    Получает временную ссылку для скачивания файла с Яндекс.Диска.
    """
    print(f"Получение временной ссылки для: {public_url}")
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {
        "public_key": public_url
    }
    
    try:
        response = requests.get(api_url, params=params)
        # Проверяем, что запрос к API прошел успешно (код 2xx)
        response.raise_for_status() 
        
        data = response.json()
        download_link = data.get("href")
        
        if not download_link:
            print("Ошибка: API Яндекса не вернул ссылку для скачивания (ключ 'href' отсутствует).")
            return None
            
        print("Временная ссылка успешно получена.")
        return download_link
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к API Яндекса: {e}")
        return None
    except ValueError: # Если ответ не в формате JSON
        print("Ошибка: не удалось разобрать JSON-ответ от API Яндекса.")
        return None

def download_file(url, dest_path):
    print(f"Скачивание из {url[:70]}...") # Обрезаем URL для читаемости
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Файл сохранён: {dest_path}")
    else:
        print(f"Ошибка скачивания! Код ответа: {response.status_code}")
        print("Возможно, срок действия временной ссылки истек или возникла другая проблема.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Скрипт для скачивания датасетов с Яндекс.Диска")
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=False,
                        help="Какой датасет скачать: original или translated. Если не указано — скачать все.")
    args = parser.parse_args()

    if args.dataset:
        datasets_to_download = [args.dataset]
    else:
        datasets_to_download = list(DATASETS.keys())

    for ds_key in datasets_to_download:
        dataset = DATASETS[ds_key]
        save_dir = dataset["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        dest_path = os.path.join(save_dir, dataset["filename"])

        if os.path.exists(dest_path):
            print(f"Файл уже существует: {dest_path}. Пропускаем.")
            continue # Переходим к следующему файлу

        # Получаем временную ссылку
        download_link = get_yandex_disk_download_link(dataset["public_url"])

        # Если ссылка получена, скачиваем файл
        if download_link:
            download_file(download_link, dest_path)
        else:
            print(f"Не удалось получить ссылку для '{ds_key}'. Скачивание отменено.")
            # Решаем, прерывать ли весь скрипт или нет. 
            # Лучше просто пропустить этот файл.
            # sys.exit(1) 

if __name__ == "__main__":
    main()