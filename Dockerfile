FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching — only reinstalls if
# requirements.txt actually changes, not on every code edit).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy application code and model artifacts.
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# --host 0.0.0.0 is required: uvicorn's default (127.0.0.1) only accepts
# connections from inside the container itself, so requests from outside
# (even via the mapped port) would fail.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]