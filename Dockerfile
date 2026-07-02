FROM python:3.11-slim

WORKDIR /app

# System deps for Prophet / pystan
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data dirs
RUN mkdir -p data/raw data/processed

EXPOSE 7860

CMD ["python", "src/dashboard/app.py"]
