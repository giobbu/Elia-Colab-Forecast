# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building Python C extensions
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container at /app
COPY . /app

# Specify the command to run the application
CMD ["python", "main.py"]
