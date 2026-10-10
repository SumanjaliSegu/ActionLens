# Hugging Face Spaces (Docker SDK) / any container host. CPU only.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    ROBOTRACE_WORK_DIR=/tmp/robotrace HOME=/tmp
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
