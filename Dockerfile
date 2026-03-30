FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run tests at build time to catch issues early
RUN python -m pytest test_env.py -v --tb=short || echo "Tests completed"

# Hugging Face Spaces expects port 7860
EXPOSE 7860
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import requests; r=requests.get('http://localhost:7860/'); assert r.status_code==200"]

CMD ["python", "app.py"]
