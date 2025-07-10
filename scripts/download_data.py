import argparse
import os
import sys
import requests

DATASETS = {
    "original": {
        "url": "https://downloader.disk.yandex.ru/disk/d6cfd449365cdc397dce0960c74e2287e6edd91a1efdb30c93a292d42f7e62b8/686d6803/ECQQcvrfp370XTII-0SDkAmsaV_d7U_uJ0t-zwJU1rt-muUDDNWUnqC8ZPYceKrDEVEPSdKtKacx8yzAla2F9g%3D%3D?uid=0&filename=customer_support_tickets.csv&disposition=attachment&hash=P847k4kWG04YWygAsy0Ak7cKpzEQ3NJFnuu2o83gS5/BNTyO0S0/5%2BYQyOcinwTrq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=text%2Fplain&owner_uid=795869371&fsize=3945533&hid=26f918f2fa4fbdb2275671f3905ed450&media_type=data&tknv=v3",
        "filename": "customer_support_tickets.csv",
        "save_dir": "data/raw"
    },
    "translated": {
        "url": "https://downloader.disk.yandex.ru/disk/81e108f339e3e0ef3a1378f4ae5e22fbebae03bdb6e7c6d350b6e56e758c1066/686d6835/i0i6Yaa7_UhYBPsu1K52dyeCRpEefwmphLTwsR2bPwKnUs6GF73FueciXx8WptDscc3aGjVdXBwJrkXSDyHtWg%3D%3D?uid=0&filename=customer_support_tickets_translated.csv&disposition=attachment&hash=xJ8vXDNTdAzzREchd0auejmw2UMQ1E2%2B4X5piMnMz0pTVMR1pr0EFCSgrZjeaYtJq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=text%2Fplain&owner_uid=795869371&fsize=8342791&hid=1f87ec8203f2906e4631eb77de3c782c&media_type=data&tknv=v3",
        "filename": "customer_support_tickets_translated.csv",
        "save_dir": "data/raw"
    }
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
            print(f"Файл уже существует: {dest_path}")
        else:
            download_file(dataset["url"], dest_path)

if __name__ == "__main__":
    main()