FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/root/.cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# preload embedding model 
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY proto /proto

RUN python -m grpc_tools.protoc \
    -I /proto \
    --python_out=. \
    --grpc_python_out=. \
    /proto/imprimer.proto

COPY . .

RUN mkdir -p data logs

EXPOSE 50051

CMD ["python", "main.py"]