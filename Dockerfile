# 1. Base Image
FROM python:3.9-slim

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

# 6. Build the Model
RUN python model/train.py

# 7. Expose the port
EXPOSE 8000

# 8. Run the Application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]