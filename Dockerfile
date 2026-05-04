FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --only=main --no-dev

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "src.api.main"]
