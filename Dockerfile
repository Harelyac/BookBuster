# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy your Python files into the container
COPY . /app

# Install dependencies if you have requirements.txt
# Remove this line if no dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run your Python script (change main.py if needed)
CMD ["python", "main.py"]
