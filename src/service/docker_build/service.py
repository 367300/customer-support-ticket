import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


MODEL_PATH = 'model.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'
STOPWORDS = set(stopwords.words('russian'))
STEMMER = SnowballStemmer('russian')


tags_metadata = [
    {
        "name": "inference",
        "description": "Классификация обращений пользователей в категории поддержки."
    },
    {
        "name": "health",
        "description": "Проверка живости сервиса."
    }
]

app = FastAPI(
    title="Support Ticket Classifier API",
    description=(
        "API сервиса классификации обращений пользователей.\n\n"
        "- Эндпоинт `/predict` принимает текст обращения и возвращает предсказанную категорию,\n"
        "  а также топ-N (по умолчанию 3) альтернативных классов с оценками.\n"
        "- Документация доступна на `/docs` (Swagger UI) и `/redoc` (ReDoc)."
    ),
    version="1.0.0",
    contact={
        "name": "Support ML Team",
        "email": "ml-team@example.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc"
)

class Query(BaseModel):
    text: str = Field(
        ..., description="Текст обращения пользователя на русском языке",
        examples=["Удалите мой аккаунт!", "Хочу изменить адрес доставки заказа"]
    )


class PredictResponse(BaseModel):
    category: str = Field(..., description="Предсказанная категория обращения")
    top_classes: Optional[List[str]] = Field(
        default=None,
        description="Список топ-N классов по убыванию уверенности модели"
    )
    top_scores: Optional[List[float]] = Field(
        default=None,
        description=(
            "Оценки decision_function для соответствующих классов (SVM). БО́льше — увереннее."
        )
    )


class HealthResponse(BaseModel):
    status: bool = Field(..., description="Признак работоспособности сервиса")

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^а-яёa-z0-9 ]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    stems = [STEMMER.stem(t) for t in tokens]
    return ' '.join(stems)

# Загрузка модели и vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:

    print('------------------------------------')
    print("Модель или векторизатор не были обнаружены, начинается обучение на лету!")
    print('------------------------------------')
    
    import subprocess

    subprocess.run([
        "python", "download_data.py",
        "--dataset", "train_data",
        "--target-dir", "data/raw"
    ], check=True)
    # Если pickle нет, обучаем на лету (для dev)
    df = pd.read_csv('data/raw/train_data.csv')
    df = df.drop_duplicates(subset=['utterance_ru']).reset_index(drop=True)
    df['utterance_ru_stemmed'] = df['utterance_ru'].apply(preprocess)
    X = df['utterance_ru_stemmed']
    y = df['request']
    vectorizer = TfidfVectorizer(max_features=10000)
    X_vec = vectorizer.fit_transform(X)
    model = LinearSVC(class_weight='balanced', random_state=42)
    model.fit(X_vec, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

@app.post(
    '/predict',
    tags=["inference"],
    summary="Классифицировать обращение пользователя",
    description=(
        "Возвращает предсказанную категорию обращения. Дополнительно возвращаются топ-3"
        " альтернативных класса с их оценками `decision_function`."
    ),
    response_model=PredictResponse,
    response_model_exclude_none=True,
    responses={
        200: {
            "description": "Успешная классификация",
            "content": {
                "application/json": {
                    "examples": {
                        "sample": {
                            "summary": "Пример ответа",
                            "value": {
                                "category": "delete_account",
                                "top_classes": [
                                    "delete_account",
                                    "change_order",
                                    "complaint"
                                ],
                                "top_scores": [
                                    1.5286598219615397,
                                    -1.0658442957804601,
                                    -1.1195215647159573
                                ]
                            }
                        }
                    }
                }
            }
        }
    }
)
def predict(query: Query) -> PredictResponse:
    text = preprocess(query.text)
    X_vec = vectorizer.transform([text])
    pred = model.predict(X_vec)[0]
    # Для SVM нет predict_proba, но можно получить decision_function
    try:
        scores = model.decision_function(X_vec)[0]
        top_classes = model.classes_[scores.argsort()[::-1][:3]]
        top_scores = scores[scores.argsort()[::-1][:3]]
        result = {
            'category': pred,
            'top_classes': list(top_classes),
            'top_scores': list(map(float, top_scores))
        }
    except Exception:
        result = {'category': pred}
    return PredictResponse(**result)

@app.get('/health', tags=["health"], summary="Проверка статуса", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port) 


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Локальный запуск"},
        {"url": "http://midas-hand.ru:5001", "description": "Продакшен"}
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi