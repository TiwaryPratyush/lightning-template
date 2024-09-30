FROM python:3.9

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install uv
RUN uv pip install -e .

COPY . .

CMD ["python", "src/train.py"]