FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Set Python path to include source
ENV PYTHONPATH=/app/src

# Default command - can be overridden
CMD ["python", "src/main.py"]
