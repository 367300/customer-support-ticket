# Customer Support Ticket Classification Service

## Описание проекта

Данный проект — это сервис для автоматической классификации обращений пользователей в техническую поддержку. На вход подаётся текст обращения, на выходе — категория обращения и ближайшие по смыслу классы. Сервис реализован на FastAPI и разворачивается в Docker-контейнере.

**Вся работа выполнена с использованием классических инструментов машинного обучения (scikit-learn, nltk и др.), без глубокого обучения и LLM. Благодаря этому получился очень быстрый микросервис: время отклика на локальном сервере составляет всего 8-9 миллисекунд.**

## Использованные датасеты

В проекте использовались два разных датасета (оба изначально на английском языке):

1. **Customer Support Ticket Dataset**  
   - Оригинал: [Kaggle](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset/data)
   - Перевод на русский: локальная нейросеть facebook/nllb-200-3.3B
   - Результат: эксперименты с этим датасетом не дали приемлемого качества (accuracy ~20%), тексты слишком шаблонные, классы неразличимы.
   - Подробнее: [notebooks/customer_support_tickets/README.md](notebooks/customer_support_tickets/README.md)

2. **Messages to Technical Support Dataset**  
   - Оригинал: [GitHub](https://github.com/BlackTeaCFG/messages-to-technical-support)
   - Перевод на русский: Яндекс.Переводчик
   - Результат: удалось построить качественную модель (accuracy > 98%) на классических ML-методах.
   - Подробнее: [notebooks/messages_to_technical_support/README.md](notebooks/messages_to_technical_support/README.md)

## Перевод датасетов
- Первый датасет переводился с помощью локальной модели facebook/nllb-200-3.3B (см. инструкции в notebooks/customer_support_tickets/README.md)
- Второй датасет переводился через Яндекс.Переводчик (API)

## Загрузка данных
Для локальных экспериментов необходимо скачать датасеты. Это делается автоматически через скрипт:

```bash
python scripts/download_data.py           # скачать все датасеты
python scripts/download_data.py --dataset original      # только первый (customer_support_tickets, оригинал)
python scripts/download_data.py --dataset translated    # только первый (customer_support_tickets, перевод)
python scripts/download_data.py --dataset train_data    # только train_data.csv (messages_to_technical_support)
python scripts/download_data.py --dataset test_data     # только test_data.csv (messages_to_technical_support)
```

Данные скачиваются с Яндекс.Диска и сохраняются в папку `data/raw/`.

## Кратко про эксперименты и выводы
- **Первый датасет** (customer_support_tickets): любые классические ML-методы показывают очень низкое качество (accuracy ~20%), причина — неинформативные тексты и отсутствие различий между классами.
- **Второй датасет** (messages_to_technical_support): после перевода и базовой обработки (стемминг, удаление стоп-слов, удаление дубликатов) классические модели (SVM+Tfidf) показывают accuracy > 98%.
- Все этапы экспериментов, EDA, обработка, обучение и тестирование подробно описаны в соответствующих ноутбуках.
- **Благодаря использованию только классических инструментов, итоговый сервис получился очень быстрым: среднее время ответа на локальном сервере — 8-9 мс.**

## Быстрый старт: как запустить сервис

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/367300/customer-support-ticket.git
   cd customer-support-ticket
   ```
2. Затем запустите сборку и запуск сервиса (необходим установленный докер)

   ```bash
   docker compose up -d
   ```

   Если возникнут ошибки, можно явно указать сборку:

   ```bash
   docker compose build
   
   # Если не указывать флаг -d то запуск будет не на фоне, а прямо в оболочке
   docker compose up
   ```
   Сервис будет доступен на http://localhost:8000 (или на порту, который вы укажете)

2. **Сделайте запрос к эндпоинту /predict:**
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "как восстановить пароль от аккаунта?"}'
   ```
   Пример ответа:
   ```json
   {
     "category": "recover_password",
     "top_classes": ["recover_password", "edit_account", "delete_account"],
     "top_scores": [12.3, 8.1, 7.5]
   }
   ```

3. **(Опционально) Переобучите модель:**
   - Если хотите обновить модель — обучите её в ноутбуке и сохраните файлы `model.joblib` и `vectorizer.joblib` в папку `src/service/`.

## Настройка порта и переменных окружения

Вы можете запускать сервис на любом удобном порту, не только на 8000. Для этого используйте переменные окружения:

- **PORT** — внутренний порт, на котором работает сервис внутри контейнера (по умолчанию 8000)
- **PORT_PREDICT** — внешний порт, на который будет проброшен сервис (по умолчанию 8000)

### Использование .env файла
Создайте файл `.env` в корне проекта:
```
PORT_PREDICT=5001
PORT=5001
```

### Пример запуска с кастомным портом

- Через .env:
  ```bash
  docker compose up
  # сервис будет доступен на http://localhost:5001
  ```
- Через переменные среды:
  ```bash
  PORT_PREDICT=5002 PORT=5002 docker compose up
  # сервис будет доступен на http://localhost:5002
  ```

В коде сервиса порт автоматически читается из переменной окружения `PORT`.

## Структура проекта
```
customer-support-ticket/
├── data/
│   └── raw/                # исходные и переведённые датасеты
├── notebooks/
│   ├── customer_support_tickets/         # эксперименты с первым датасетом
│   └── messages_to_technical_support/    # эксперименты со вторым датасетом
├── scripts/
│   └── download_data.py    # скрипт для загрузки данных
├── src/
│   └── service/            # код сервиса, Dockerfile, модель
├── docker-compose.yml
└── README.md               # этот файл
```

## Полезные ссылки
- [Описание и анализ первого датасета (customer_support_tickets)](notebooks/customer_support_tickets/README.md)
- [Описание и анализ второго датасета (messages_to_technical_support)](notebooks/messages_to_technical_support/README.md)

---

> **Проект позволяет быстро развернуть сервис для автоматической классификации обращений пользователей в техподдержку, а также воспроизвести все эксперименты и дообучить модель под свои задачи.**

