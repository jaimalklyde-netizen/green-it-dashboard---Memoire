FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn joblib pandas scikit-learn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]