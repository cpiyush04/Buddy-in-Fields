FROM python:3.11.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.lock and install Python dependencies
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock

# Copy application code
COPY . .

# Create necessary directories for uploads and data
RUN mkdir -p uploads/backend uploads/frontend uploads/speech data/qdrant_db data/docs_db data/parsed_docs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main_assistant:app", "--host", "0.0.0.0", "--port", "8000"]
