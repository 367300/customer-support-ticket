import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.pipelines import pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
from tqdm import tqdm
import os
import tempfile
import shutil

# Параметры
MODEL_PATH = './NLLB-200-3.3B-AWQ'  # Путь к скачанной модели
SOURCE_LANG = 'eng_Latn'
TARGET_LANG = 'rus_Cyrl'
BATCH_SIZE = 300
INPUT_CSV = 'data/raw/customer_support_tickets.csv'
OUTPUT_CSV = 'data/processed/customer_support_tickets_translated.csv'
TEXT_COLUMN = 'Ticket Description'
TRANSLATED_COLUMN = TEXT_COLUMN + '_ru'
ID_COLUMN = 'Ticket ID'  # Уникальный идентификатор строки

print('Загрузка токенизатора и модели...')
model_name = 'facebook/nllb-200-3.3B'
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype='auto',
    device_map='auto'
)

print('Создание pipeline...')
translator = pipeline(
    'translation',
    model=model,
    tokenizer=tokenizer,
    src_lang=SOURCE_LANG,
    tgt_lang=TARGET_LANG,
)

print('Загрузка исходного датасета...')
df = pd.read_csv(INPUT_CSV)
if ID_COLUMN not in df.columns:
    df[ID_COLUMN] = df.index  # если нет уникального ID, используем индекс

# Загружаем файл перевода, если он есть
if os.path.exists(OUTPUT_CSV):
    df_out = pd.read_csv(OUTPUT_CSV)
    if ID_COLUMN not in df_out.columns:
        df_out[ID_COLUMN] = df_out.index
    # Мержим по ID, чтобы не потерять новые строки
    df_merged = pd.merge(df, df_out[[ID_COLUMN, TRANSLATED_COLUMN]] if TRANSLATED_COLUMN in df_out.columns else df_out[[ID_COLUMN]],
                         on=ID_COLUMN, how='left')
    if TRANSLATED_COLUMN not in df_merged.columns:
        df_merged[TRANSLATED_COLUMN] = ''
    print(f'Найдено уже переведённых строк: {df_merged[TRANSLATED_COLUMN].notna().sum()} из {len(df_merged)}')
else:
    df_merged = df.copy()
    df_merged[TRANSLATED_COLUMN] = ''

texts = df_merged[TEXT_COLUMN].fillna('').astype(str).tolist()
translated = df_merged[TRANSLATED_COLUMN].fillna('').astype(str).tolist()

print('Перевод...')
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_indices = list(range(i, min(i+BATCH_SIZE, len(texts))))
    # Пропускаем уже переведённые строки
    if all(translated[j] and str(translated[j]).strip() for j in batch_indices):
        continue
    batch = [texts[j] for j in batch_indices]
    outputs = translator(batch, max_length=512)
    for idx, out in zip(batch_indices, outputs):
        df_merged.at[idx, TRANSLATED_COLUMN] = out['translation_text']
        translated[idx] = out['translation_text']
    # Сохраняем после каждого батча во временный файл, затем атомарно переименовываем
    with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(OUTPUT_CSV), suffix='.tmp') as tmpfile:
        df_merged.to_csv(tmpfile.name, index=False)
        tmpfile.flush()
        os.fsync(tmpfile.fileno())
    shutil.move(tmpfile.name, OUTPUT_CSV)

print(f'Готово! Переведённый датасет сохранён в {OUTPUT_CSV}')