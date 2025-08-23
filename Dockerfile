FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file 
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app will run on (Hugging Face default is 7860)
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]