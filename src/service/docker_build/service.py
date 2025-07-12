import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
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



app = FastAPI()

class Query(BaseModel):
    text: str

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

@app.post('/predict')
def predict(query: Query):
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
    return result

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port) 