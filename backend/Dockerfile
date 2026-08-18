FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch/torchvision first (much smaller + faster build than
# the default CUDA wheels, and Hugging Face's free Space tier is CPU-only).
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

# Hugging Face Spaces route external traffic to port 7860 by default.
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
