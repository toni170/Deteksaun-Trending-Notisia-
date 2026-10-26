# Dockerfile — Mining Notísia Online no Deteksaun Topiku
# Build : docker build -t tatoli-topic-analyzer .
# Run   : docker run -p 8501:8501 -v $(pwd)/data:/app/data tatoli-topic-analyzer
#
# Catatan: volume -v $(pwd)/data:/app/data WAJIB dipakai kalau ingin data
# (data/raw, data/cleaned, dst) tetap tersimpan setelah container di-restart.
# Tanpa volume, data akan hilang setiap kali container dibuat ulang.

FROM python:3.11-slim

WORKDIR /app

# Dependency sistem yang dibutuhkan lxml, matplotlib, dan wordcloud
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/cleaned data/models data/keywords data/batches data/uploads logs

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/dashboard/trend_dashboard.py", \
    "--server.port=8501", "--server.address=0.0.0.0"]
