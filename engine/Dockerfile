FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY --from=proto . /proto
RUN python -m grpc_tools.protoc \
    -I /proto \
    --python_out=. \
    --grpc_python_out=. \
    /proto/imprimer.proto

RUN mkdir -p data logs

EXPOSE 50051

CMD ["python", "main.py"]