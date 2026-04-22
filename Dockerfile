FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py sim.py env.py server.py trajectory_logger.py ./

CMD ["python", "server.py"]
