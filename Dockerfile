# Trading Bot Dockerfile
# ======================
# Production-ready container for 24/7 trading bot execution

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements_bot.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_bot.txt

# Copy application code
COPY bot/ ./bot/
COPY reports/ ./reports/
COPY notifications/ ./notifications/
COPY deployment/ ./deployment/
COPY data_providers/ ./data_providers/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash botuser && \
    chown -R botuser:botuser /app

USER botuser

# Default command
CMD ["python", "-m", "bot.main"]

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=30s --retries=3 \
    CMD python -c "from bot.trader import EnhancedTrader; t = EnhancedTrader(); print('OK' if t.is_connected else 'FAIL')" || exit 1
