FROM golang:1.25.6-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY gateway/ ./gateway/

WORKDIR /app/gateway
RUN go build -o /imprimer-gateway ./cmd/main.go

FROM alpine:latest
COPY --from=builder /imprimer-gateway /imprimer-gateway
EXPOSE 8080
CMD ["/imprimer-gateway"]