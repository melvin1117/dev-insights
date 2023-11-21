# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Copy the current directory contents into the container at /usr/src/app
COPY ./requirements.txt /requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory in the container
WORKDIR /app
