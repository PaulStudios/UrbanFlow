# Use the official Python image as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /mock_api

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Install Alembic and initialize the database
RUN pip install alembic

# Expose the port the app runs on
EXPOSE 8001

# Initialize the database using Alembic and run the application
CMD alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000