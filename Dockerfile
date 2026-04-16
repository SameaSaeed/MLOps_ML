FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/

RUN apt-get update && \
    apt-get install -y libpq-dev build-essential && \
    python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
COPY models/ /app/models/

# Set working directory to src
WORKDIR /app/src

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI server using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
