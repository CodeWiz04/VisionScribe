FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching — only reinstalls if
# requirements.txt actually changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy application code and model artifacts.
# NOTE: src/ is included alongside app/ because app/inference.py imports
# decoder.py and preprocessing.py from there (see SRC_DIR in inference.py).
# The original brief's Dockerfile only copied app/ and models/, which
# would fail at import time since decoder.py/preprocessing.py live in src/.
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# --host 0.0.0.0 is required, not optional: uvicorn's default host
# (127.0.0.1) only accepts connections from inside the container itself,
# so requests from outside via the mapped port would fail.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]