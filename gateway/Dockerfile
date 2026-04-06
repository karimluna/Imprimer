FROM golang:1.26.1-alpine AS builder

WORKDIR /app

# go.mod is at root — context is project root
COPY go.mod go.sum ./
RUN go mod download

# Copy gateway source
COPY gateway/ ./gateway/

# Proto passed as additional context
COPY --from=proto . /proto

RUN apk add --no-cache protobuf && \
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest && \
    mkdir -p gateway/gen && \
    protoc -I /proto \
      --go_out=gateway/gen \
      --go-grpc_out=gateway/gen \
      --go_opt=paths=source_relative \
      --go-grpc_opt=paths=source_relative \
      /proto/imprimer.proto

RUN go build -o /imprimer-gateway ./gateway/cmd/main.go

FROM alpine:latest
COPY --from=builder /imprimer-gateway /imprimer-gateway
EXPOSE 8080
CMD ["/imprimer-gateway"]