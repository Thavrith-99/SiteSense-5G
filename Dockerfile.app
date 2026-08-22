# SiteSense 5G — Streamlit dashboard image (also used for the one-shot DB loader).
# Python 3.12 (leafmap/rasterio/streamlit all have 3.12 wheels; TF is NOT needed here).
FROM python:3.12-slim

WORKDIR /app

# curl for the container healthcheck; rasterio ships GDAL in its wheel so no apt GDAL needed.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
