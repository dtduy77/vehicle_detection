# Use an official Python runtime as a parent image
FROM python:3.11.1

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the font file into the image
# COPY arial.ttf /usr/share/fonts/truetype/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port available to the world outside this container
EXPOSE 8001

# Run your FastAPI app
CMD ["uvicorn", "app:app", "--reload", "--port=8001", "--host=0.0.0.0"]


