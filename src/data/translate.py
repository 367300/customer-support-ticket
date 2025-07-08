import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, BitsAndBytesConfig
from tqdm import tqdm

# Путь к локальной квантизированной модели (AWQ)
MODEL_PATH = './NLLB-200-3.3B-AWQ'  # Укажи путь к скачанной модели
SOURCE_LANG = 'eng_Latn'
TARGET_LANG = 'rus_Cyrl'
BATCH_SIZE = 24  # Можно увеличить, если хватает памяти
INPUT_CSV = 'customer_support_tickets.csv'
OUTPUT_CSV = 'customer_support_tickets_translated.csv'
TEXT_COLUMN = 'Ticket Description'

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

print('Загрузка данных...')
df = pd.read_csv(INPUT_CSV)
texts = df[TEXT_COLUMN].fillna('').astype(str).tolist()

print('Перевод...')
translated = []
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i+BATCH_SIZE]
    outputs = translator(batch, max_length=400)
    translated.extend([o['translation_text'] for o in outputs])

print('Сохраняю результат...')
df[TEXT_COLUMN + '_ru'] = translated
df.to_csv(OUTPUT_CSV, index=False)
print(f'Готово! Переведённый датасет сохранён в {OUTPUT_CSV}') 