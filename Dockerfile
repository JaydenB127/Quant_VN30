# Use stable python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv package manager for extremely fast builds
RUN pip install uv

# Copy package configurations
COPY pyproject.toml setup.py /app/

# Install python dependencies using uv
RUN uv pip install --system -e . && uv pip install --system -e ".[dev]"

# Copy application code
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "ets.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
