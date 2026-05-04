FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY pyproject.toml ./

# Install dependencies using pip
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.5.0 \
    pandas \
    numpy \
    scikit-learn \
    catboost>=1.2.10 \
    xgboost>=3.2.0 \
    lightgbm>=4.6.0 \
    imbalanced-learn

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "src.api.main"]
