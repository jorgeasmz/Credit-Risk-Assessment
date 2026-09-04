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

# 6. Fetch the model the registry pinned. Training here would ship a model that
# nothing measured and nothing compared against the one it replaces, and would tie
# the build to the dataset's host staying up. The pointer names a commit, so the
# image gets exactly the bytes the gate scored, over public HTTPS and with no
# credential in the build.
RUN python -m model.fetch

# 7. Expose the port
EXPOSE 8000

# 8. Migrate, then serve.
# Migrations run at startup rather than at build time because the database does
# not exist yet when the image is built. Render and most container hosts inject
# $PORT and expect the process to bind to it; exec form would not expand the
# variable, so this goes through sh, and "exec" keeps uvicorn as PID 1 so it
# still receives SIGTERM.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
