# Use a slim Python image as the base. This provides Python and a minimal Linux environment.
# python:3.9-slim-buster is a good choice for smaller image sizes.
FROM python:3.10-slim-buster

# Set environment variables for Python to prevent buffering of stdout/stderr.
# This makes logs from inside the container appear immediately in the console.
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
# All subsequent commands will be executed relative to this directory.
WORKDIR /app

# Copy the requirements.txt file into the container's /app directory.
# This step is done early to leverage Docker's build cache. If only requirements.txt changes,
# Docker won't re-run the pip install command.
COPY requirements.txt /app/

# Install system dependencies required for psycopg2-binary and other Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6 \
    libjpeg-dev \
    zlib1g-dev \
    libportaudio2 \
    portaudio19-dev \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt.
# --no-cache-dir: Prevents pip from storing downloaded packages in a cache,
#                 reducing the final image size.
RUN pip install --no-cache-dir --verbose -r requirements.txt
# Copy the entire Django project into the container's /app directory.
# This copies manage.py, myproject/, myapp/, etc.
# Note: This is done after installing dependencies to again leverage build cache.
COPY . /app/

# Collect static files into the STATIC_ROOT directory defined in settings.py.
# --noinput: Prevents prompts during the collection process (e.g., "Are you sure?").
# This is crucial for serving static assets correctly in production.
RUN python manage.py collectstatic --noinput

# Expose port 8000. This informs Docker that the container listens on this port at runtime.
# It doesn't actually publish the port; it's more for documentation and networking configuration.
EXPOSE 8000

# Define the command to run when the container starts.
# We're using Gunicorn (a Python WSGI HTTP Server) to serve the Django application.
# "myproject.wsgi:application" refers to your Django project's WSGI application.
# "--bind 0.0.0.0:8000": Binds Gunicorn to all network interfaces on port 8000 inside the container.
# This makes the application accessible from outside the container.
CMD ["gunicorn", "myproject.wsgi:application", "--bind", "0.0.0.0:8000"]
