import pandas as pd
import os

# --- 1. Определяем пути к файлам ---
main_data_path = 'data/raw/train_data.csv'
translated_utterance_path = 'data/raw/train_data_utterance.csv'

# Директория для сохранения и имя конечного файла
output_dir = 'data/processed'
output_file_path = os.path.join(output_dir, 'train_data_full_translated.csv')

# Создаем директорию для результата, если ее нет
os.makedirs(output_dir, exist_ok=True)


# --- 2. Загружаем основной датасет ---
# Используем разделитель '|', так как в твоем примере именно он
print(f"Загрузка основного файла: {main_data_path}")
try:
    df_main = pd.read_csv(main_data_path, sep='|')
    print("Основной датасет успешно загружен. Первые 5 строк:")
    print(df_main.head())
except FileNotFoundError:
    print(f"Ошибка: файл не найден по пути '{main_data_path}'")
    exit() # Прерываем выполнение, если основного файла нет

# --- 3. Загружаем файл с переведенными фразами ---
# pandas по умолчанию правильно обработает кавычки и уберет их
print(f"\nЗагрузка файла с переводом: {translated_utterance_path}")
try:
    # Загружаем только один столбец и сразу получаем его как pd.Series
    translated_series = pd.read_csv(translated_utterance_path)['utterance']
    print("Переведенный столбец успешно загружен. Первые 5 строк:")
    print(translated_series.head())
except FileNotFoundError:
    print(f"Ошибка: файл не найден по пути '{translated_utterance_path}'")
    exit() # Прерываем выполнение, если файла с переводом нет


# --- 4. Объединяем данные ---
# Проверяем, что количество строк совпадает для корректного объединения
if len(df_main) != len(translated_series):
    print("\nОшибка: количество строк в основном файле и файле с переводом не совпадает!")
    print(f"Строк в основном файле: {len(df_main)}")
    print(f"Строк в файле с переводом: {len(translated_series)}")
    exit()

# Добавляем переведенный столбец в основной DataFrame.
# Дадим ему понятное имя, например, 'utterance_ru'
df_main['utterance_ru'] = translated_series

print("\nДатасеты успешно объединены. Результат:")
print(df_main.head())


# --- 5. Сохраняем итоговый датасет в новый CSV файл ---
# Будем использовать стандартный разделитель - запятую
print(f"\nСохранение итогового файла в: {output_file_path}")
df_main.to_csv(
    output_file_path, 
    index=False,         # Не сохраняем индекс pandas в файл
    encoding='utf-8'     # Используем utf-8 для корректной работы с кириллицей
)

print("Готово! Итоговый файл успешно сохранен.")