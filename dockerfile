FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libxfixes3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pdf_outline_extractor.py .
COPY challenge_1b.py .

# Create input/output folders (required by execution)
RUN mkdir -p /app/input /app/output

# Set entrypoint to run challenge_1b.py with user-supplied args
ENTRYPOINT ["python", "challenge_1b.py"]
