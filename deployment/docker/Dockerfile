FROM python:3.12-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 python3-tk \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . /app
RUN pip install --no-cache-dir -e ".[dev,legacy,monitoring]"
ENV MPLBACKEND=Agg PYTHONUTF8=1
CMD ["python", "-m", "pytest"]
