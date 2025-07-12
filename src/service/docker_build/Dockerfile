FROM python:3.12-slim
WORKDIR /app/service
COPY . /app/service
COPY download_data.py /app/service/download_data.py
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir fastapi uvicorn scikit-learn pandas nltk joblib requests
# Для nltk stopwords
RUN python -m nltk.downloader stopwords
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]