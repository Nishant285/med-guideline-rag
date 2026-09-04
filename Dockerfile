# Base image: slim Python, keeps the container small and the build fast.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, separately from copying the rest of the code.
# Docker caches this layer, so re-deploys after a small code change don't
# reinstall every package from scratch - only the final COPY layer changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and the pre-built search index.
COPY . .

# Hugging Face Spaces expects the app to listen on port 7860 by default.
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
