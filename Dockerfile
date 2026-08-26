# 1. Base Image
FROM python:3.12-slim

# 2. Set Working Directory inside the container
WORKDIR /app

# 3. Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY . .

# 6. Build the Model. Run as a module: train.py imports from the model package.
RUN python -m model.train

# 7. Expose the port
EXPOSE 8000

# 8. Run the Application.
# Render and most container hosts inject $PORT and expect the process to bind
# to it. Exec form would not expand the variable, so this goes through sh, and
# "exec" hands PID 1 to uvicorn so it still receives SIGTERM on shutdown.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
