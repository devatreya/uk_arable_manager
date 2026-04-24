FROM python:3.11 AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY config.py sim.py env.py server.py trajectory_logger.py ./
COPY data/processed ./data/processed

ENV PATH=/root/.local/bin:$PATH

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
