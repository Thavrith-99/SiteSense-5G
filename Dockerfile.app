# SiteSense 5G — Streamlit dashboard image (also used for the one-shot DB loader).
# Full python:3.12 (NOT slim): rasterio's bundled GDAL needs base system libs like
# libexpat.so.1 that slim omits (slim caused "ImportError: libexpat.so.1" at page load).
# The API image uses this same full base and runs rasterio fine.
FROM python:3.12

WORKDIR /app

# curl for the container healthcheck (rasterio ships GDAL in its wheel; the full base has the rest).
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
