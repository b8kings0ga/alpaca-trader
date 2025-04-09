FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=America/New_York

# Set working directory
WORKDIR /app

# Install all system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/logs /app/models

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make main.py executable
RUN chmod +x main.py

# Add crontab file for more frequent trading (every 15 minutes during market hours)
RUN echo "*/15 9-16 * * 1-5 cd /app && python main.py >> /app/logs/cron.log 2>&1" > /etc/cron.d/alpaca-trader \
    && chmod 0644 /etc/cron.d/alpaca-trader \
    && crontab /etc/cron.d/alpaca-trader

# Create entrypoint script
RUN echo '#!/bin/sh\n\
echo "Starting Alpaca Trading Bot..."\n\
cron\n\
echo "Running initial trading cycle..."\n\
python main.py\n\
echo "Bot is now scheduled to run every 15 minutes during market hours. Keeping container alive..."\n\
tail -f /app/logs/alpaca_trader.log\n\
' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]