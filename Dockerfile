FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway ./gateway
COPY inspect_space.py ./inspect_space.py
RUN useradd -m -u 1000 appuser
USER appuser
EXPOSE 7860
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
