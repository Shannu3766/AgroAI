FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT 8080
EXPOSE 8080
ENV FLASK_APP=app.py
CMD gunicorn --bind "0.0.0.0:$PORT" --workers 1 --threads 8 --timeout 0 app:app