# Описание датасета

**Customer Support Ticket Dataset** — это набор данных, включающий обращения в службу поддержки клиентов по различным технологическим продуктам. В нем содержатся запросы клиентов, связанные с аппаратными проблемами, программными ошибками, сетевыми сбоями, доступом к аккаунту, потерей данных и другими вопросами поддержки. Датасет предоставляет информацию о клиенте, приобретённом продукте, типе обращения, канале обращения, статусе заявки и других важных деталях.

Данный датасет может быть использован для различных аналитических и моделирующих задач в сфере клиентского сервиса.

## Описание признаков

- **Ticket ID**: Уникальный идентификатор каждой заявки
- **Customer Name**: Имя клиента, создавшего заявку
- **Customer Email**: Электронная почта клиента (домен @example.com используется для защиты персональных данных)
- **Customer Age**: Возраст клиента
- **Customer Gender**: Пол клиента
- **Product Purchased**: Приобретённый клиентом продукт
- **Date of Purchase**: Дата покупки продукта
- **Ticket Type**: Тип заявки (например, техническая проблема, вопрос по оплате, запрос по продукту)
- **Ticket Subject**: Тема/предмет обращения
- **Ticket Description**: Описание проблемы или вопроса клиента
- **Ticket Status**: Статус заявки (например, открыта, закрыта, ожидает ответа клиента)
- **Resolution**: Решение или предоставленное решение для закрытых заявок
- **Ticket Priority**: Приоритет заявки (например, низкий, средний, высокий, критический)
- **Ticket Channel**: Канал, через который была создана заявка (например, email, телефон, чат, социальные сети)
- **First Response Time**: Время до первого ответа клиенту
- **Time to Resolution**: Время, затраченное на решение заявки
- **Customer Satisfaction Rating**: Оценка удовлетворённости клиента для закрытых заявок (по шкале от 1 до 5)

---

## Примеры использования датасета

- **Анализ поддержки клиентов**: Анализировать тенденции обращений, выявлять частые проблемы и улучшать процессы поддержки.
- **Обработка естественного языка (NLP)**: Использовать описания заявок для обучения моделей автоматической категоризации или анализа тональности.
- **Прогнозирование удовлетворённости клиентов**: Обучать модели для предсказания уровня удовлетворённости на основе информации о заявках.
- **Прогнозирование времени решения заявки**: Строить модели для оценки времени, необходимого для решения заявки, исходя из различных факторов.
- **Сегментация клиентов**: Сегментировать клиентов по типам обращений, проблемам или уровню удовлетворённости.
- **Рекомендательные системы**: Создавать системы рекомендаций для предложения релевантных решений или продуктов на основе запросов клиентов.

---

## Перевод датасета на русский с помощью LLM (NLLB-200)

### Требования
- Python 3.8+
- NVIDIA GPU (рекомендуется 16+ ГБ видеопамяти)
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- pip install transformers pandas tqdm

### Использование

1. Установите зависимости:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers pandas tqdm
   ```
2. Запустите перевод:
   ```bash
   python translate.py
   ```
3. Результат будет сохранён в файл `customer_support_tickets_translated.csv` (добавится столбец с переводом на русский).

---

## Источник данных

Данная работа основана на открытом датасете [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset/data).

## Получение данных

Для воспроизводимости экспериментов предусмотрен автоматический скрипт загрузки данных.  
Если исходный датасет отсутствует в папке `data/raw/`, его можно скачать автоматически:

- Исходный датасет будет скачан по прямой ссылке в https://disk.yandex.ru/d/GbebgnMOzkUD_Q
- Переведённый датасет будет скачан по прямой ссылке в https://disk.yandex.ru/d/9GNABDyrbm58Jg

### Автоматическая загрузка датасетов

Для скачивания исходного или переведённого датасета используйте скрипт:

```bash
python scripts/download_data.py --dataset original      # скачать только оригинальный датасет
python scripts/download_data.py --dataset translated   # скачать только переведённый датасет
python scripts/download_data.py                       # скачать оба датасета сразу
```

- Оригинальный и переведённый датасет сохраняются в папку `data/raw/`

_Структура папок:_
- `data/raw/` — исходные (оригинальные и переведённые) датасеты
- `data/processed/` — обработанные данные

---

## Итоги экспериментов по классификации (2024)

### Цель
Проверить, можно ли добиться высокой точности (≥90%) в задаче многоклассовой классификации типа обращения (`ticket_type`) по тексту обращения и дополнительным признакам с помощью классических методов машинного обучения.

### Краткие выводы
- Были опробованы все классические ML-методы: Logistic Regression, SVM, RandomForest, а также различные способы обработки текста (TF-IDF, CountVectorizer, fastText-эмбеддинги), добавление и усиление категориальных признаков (`product_purchased`, `ticket_subject`, `ticket_channel`), подстановка реального продукта, удаление шаблонных и частых слов, удаление первых слов и др.
- Ни один из подходов не позволил добиться точности выше 22%. Результаты всех моделей — около 18–22% accuracy.
- Причина — крайне шаблонные и неинформативные тексты обращений, а также отсутствие различающих признаков между классами.
- Для повышения качества необходимы либо более информативные данные, либо использование современных языковых моделей (BERT, LLM и др.).

---

> **Данный датасет отлично подходит для задач анализа клиентской поддержки, построения моделей машинного обучения и совершенствования сервисов поддержки пользователей, однако для задач автоматической классификации типа обращения требуется более разнообразная и информативная текстовая база.**