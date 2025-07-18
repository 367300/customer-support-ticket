## Техническое задание (краткая версия по ГОСТ 34)

### **Наименование системы**

* **Полное:** Система автоматической классификации и обработки обращений пользователей.
* **Краткое:** Система "САК-О".

### 1. Общие сведения

* **Заказчик:** (указать наименование)
* **Исполнитель:** (указать наименование)
* **Основание для разработки:** Инициативная разработка с целью автоматизации процессов технической поддержки и создания учебного стенда.

### 2. Назначение и цели создания системы

#### **2.1. Назначение системы**
Система предназначена для автоматизации процесса классификации текстовых обращений пользователей в службу технической поддержки. Она также предоставляет веб-интерфейс для демонстрации работы фоновых задач, взаимодействия с ML-сервисом и отслеживания статусов в режиме реального времени.

#### **2.2. Цели создания системы**
* **Снижение** ручной нагрузки на операторов техподдержки за счет автоматического определения категории обращения.
* **Повышение скорости** обработки входящих запросов.
* **Создание** прототипа и учебного пособия для демонстрации интеграции Django-приложения с внешним ML-сервисом с использованием современных технологий (Docker, Celery, WebSocket).
* **Апробация** моделей классического машинного обучения для задачи классификации текстов на русском языке.

### 3. Характеристика объектов автоматизации

Объектом автоматизации является процесс обработки входящих сообщений от пользователей. Процесс включает прием сообщения через веб-интерфейс, отправку на классификацию в ML-сервис, получение категории и отображение результата пользователю в режиме реального времени.

### 4. Требования к системе

#### **4.1. Состав системы**
Система состоит из двух ключевых взаимодействующих подсистем, разворачиваемых в Docker-контейнерах:

1.  **Подсистема 1: Сервис классификации обращений (ML-сервис)**
    * Реализован на Python с использованием фреймворка FastAPI.
    * Использует модель классического машинного обучения (SVM+Tfidf) для определения категории текста.
    * Предоставляет REST API эндпоинт `/predict` для получения классификации.
    * Отличается высокой производительностью (время отклика ~8-9 мс, 20-40 мс на удаленном сервере).

2.  **Подсистема 2: Веб-портал обработки задач (Django-портал)**
    * Реализован на Python с использованием фреймворка Django.
    * Использует Celery и RabbitMQ для асинхронной обработки задач (отправка запросов к ML-сервису).
    * Использует Django Channels для организации WebSocket-соединений и обновления данных в реальном времени.
    * Предоставляет веб-интерфейс с чатом для отправки сообщений и отслеживания их статуса.
    * Включает административную панель Django для управления задачами и сообщениями.

#### **4.2. Функциональные требования**
* **ML-сервис** должен принимать на вход JSON с текстом обращения и возвращать JSON с наиболее вероятной категорией, списком топ-3 категорий и их оценками.
* **Django-портал** должен предоставлять пользователю интерфейс для отправки текстовых сообщений без перезагрузки страницы (AJAX).
* После отправки сообщение должно асинхронно отправляться в **ML-сервис** для классификации.
* Категория обработанного сообщения должна отображаться в интерфейсе в реальном времени через WebSocket.
* Система должна корректно обрабатывать ситуацию, когда **ML-сервис** недоступен (сообщение сохраняется без категории).

#### **4.3. Нефункциональные требования**
* **Производительность:** Время ответа ML-сервиса на запрос классификации не должно превышать 50 мс при штатной нагрузке.
* **Надежность:** Веб-портал должен оставаться работоспособным даже при временной недоступности ML-сервиса.
* **Развертывание:** Вся система должна легко разворачиваться и запускаться одной командой `docker compose up`.
* **Конфигурируемость:** Порты и другие ключевые параметры должны настраиваться через переменные окружения (`.env` файл).

### 5. Состав и содержание работ по созданию системы

1.  Проведение исследований и выбор набора данных для обучения модели.
2.  Разработка и обучение модели машинного обучения.
3.  Разработка ML-сервиса на FastAPI.
4.  Разработка базового шаблона Django-проекта с поддержкой асинхронных задач и WebSocket.
5.  Интеграция Django-портала с ML-сервисом.
6.  Тестирование комплексного взаимодействия подсистем.
7.  Подготовка документации и инструкций по развертыванию.

### 6. Порядок контроля и приемки

Приемка системы производится путем демонстрации выполнения следующих сценариев:
1.  Успешный запуск всей системы командой `docker compose up`.
2.  Отправка тестового запроса на эндпоинт ML-сервиса (`/predict`) и получение корректного ответа.
3.  Открытие страницы чата в Django-портале, отправка сообщения и автоматическое появление категории сообщения через несколько секунд.
4.  Демонстрация работы административной панели Django и мониторинга задач в Flower.

---

## Оценка проекта по методу COCOMO II

### 1. Определение размера проекта (Size)

COCOMO II использует в качестве метрики тысячи строк исходного кода (**KSLOC** - Kilo Source Lines of Code):

* **ML-сервис (FastAPI):** Небольшой микросервис с одним эндпоинтом, загрузкой модели и логикой предсказания. **Оценка: 0.3 KSLOC** (300 строк кода).
* **Django-портал:** Более сложный проект. Включает настройки Django, модели, views, Celery tasks, Channels consumers, шаблоны, Docker-файлы. **Оценка: 1.2 KSLOC** (1200 строк кода).

**Итоговый размер проекта (Size):** 0.3 + 1.2 = **1.5 KSLOC**.

### 2. Определение факторов масштаба (Scale Factors)

Это 5 факторов, которые определяют, как размер проекта влияет на трудоемкость.

| Фактор | Описание | Оценка | Обоснование |
| :--- | :--- | :--- | :--- |
| **PREC** | Новизна проекта | **High (Высокая)** | Технологии (Django, FastAPI, Docker) и задача (классификация) хорошо известны. |
| **FLEX** | Гибкость разработки | **High (Высокая)** | Проект учебный/личный, нет жестких рамок и требований от заказчика. |
| **RESL** | Разрешение рисков | **Nominal (Среднее)** | Риски (качество датасета) были выявлены и решены на раннем этапе, но процесс неформальный. |
| **TEAM** | Сработанность команды | **Very High (Очень высокая)** | Проект выполнен одним человеком, идеальная координация. |
| **PMAT** | Зрелость процессов | **Low (Низкая)** | Процессы разработки не формализованы, что типично для личных проектов. |

Эти факторы дают нам показатель степени **E = 1.059**.

### 3. Определение множителей трудоемкости (Cost Drivers)

Это 17 факторов, которые корректируют итоговую оценку. Взяты ключевые из них.

| Фактор | Описание | Оценка | Обоснование |
| :--- | :--- | :--- | :--- |
| **RELY** | Требуемая надежность | **Low (Низкая)** | Сбой в учебном проекте не критичен. |
| **CPLX** | Сложность продукта | **Nominal (Средняя)** | Логика простая (классический ML), но интеграция двух сервисов добавляет сложности. |
| **DOCU** | Документированность | **Low (Низкая)** | Есть хорошие `README`, но это не полный комплект проектной документации. |
| **ACAP** | Способности аналитика | **High (Высокие)** | Разработчик сам проанализировал проблему, нашел и решил ее (смена датасета). |
| **PCAP** | Способности программиста | **High (Высокие)** | Успешная реализация на современном стеке подтверждает высокий уровень. |
| **APEX** | Опыт в приложении | **Very High (Очень высокий)** | Судя по всему, разработчик хорошо знаком с предметной областью. |
| **TOOL** | Использование инструментов | **Very High (Очень высокий)** | Используется самый современный и продуктивный инструментарий (Docker, FastAPI, Celery). |
| **SCED** | Сжатость графика | **Nominal (Средний)** | Принимаем стандартный график. |

Произведение этих множителей дает итоговый коэффициент трудоемкости **EAF (Effort Adjustment Factor) ≈ 0.45**.

### 4. Итоговый расчет по COCOMO II

* **Трудоемкость (Effort):**
    $$PM = EAF \times A \times (Size)^E$$
    где A = 2.94 (стандартный коэффициент для COCOMO II).
    $$PM = 0.45 \times 2.94 \times (1.5)^{1.059} \approx 1.323 \times 1.55 \approx \textbf{2.05 чел.-мес.}$$

* **Время разработки (Duration):**
    $$TDEV = C \times (PM)^F$$
    где C = 3.67, а F = 0.31 (рассчитывается на основе E).
    $$TDEV = 3.67 \times (2.05)^{0.31} \approx 3.67 \times 1.24 \approx \textbf{4.55 мес.}$$

### 5. Выводы и сравнение

| Метрика | Ваш расчет (Basic COCOMO) | Мой расчет (COCOMO II) | Реальные данные |
| :--- | :--- | :--- | :--- |
| **Трудоемкость** | 3.7 чел.-мес. | **2.05 чел.-мес.** | ~1 чел.-мес. (20 дней) |
| **Длительность** | 3.9 мес. | **4.55 мес.** | ~1 мес. (20 дней) |

**Анализ результатов:**

1.  **COCOMO II дает более точную оценку трудоемкости (2.05 чел.-мес.)**, чем Basic COCOMO, так как лучше учитывает факторы высокой квалификации разработчика и использование современных инструментов. Это уже гораздо ближе к реальному 1 месяцу.
2.  **Почему оценка все равно выше реальной?** Модель COCOMO предполагает "стандартный" рабочий процесс (8-часовой день, совещания, документирование и т.д.). Данный проект, был выполнен одним высокомотивированным специалистом в режиме "спринта", с минимальными накладными расходами. Реализация за 20 дней при оценке в 2.05 чел.-месяца (~44 рабочих дня) означает, что его **персональная производительность была примерно в 2 раза выше**, чем у "среднего" высококвалифицированного разработчика по модели. Это абсолютно реально для эксперта на знакомой задаче.
3.  **Длительность в 4.55 месяца** модель рассчитывает для гипотетической команды, которая бы выполняла этот объем работ. Для одного человека трудоемкость и длительность практически совпадают (2.05 чел.-мес. ≈ 2 месяца работы).

**Итоговый вывод:** Модель COCOMO II, в отличие от базовой версии, дала вполне адекватную оценку, показав, что проект требует около **2 человеко-месяцев** стандартной работы. Ваше фактическое выполнение за 20 дней демонстрирует чрезвычайно высокую личную эффективность.