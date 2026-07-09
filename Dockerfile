FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[ui]"

COPY app.py ./
COPY data ./data
COPY scripts ./scripts
RUN python scripts/create_sample_db.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
