package main

import (
	"log"
	"net/http"

	"github.com/BalorLC3/Imprimer/gateway/internal/client"
	"github.com/BalorLC3/Imprimer/gateway/internal/handler"
	"github.com/BalorLC3/Imprimer/gateway/internal/middleware"
)

// Imprimer gateway; entry point for all external requests
// In Minsky's Society of Mind framing, this is the receptor layer:
// it receives stimuli from the outside world and routes them inward.
// It knows nothing about how prompts work or how models think.
// It only knows how to receive, authenticate, audit, and forward.

func main() {
	// Connect to Python gRPC server
	engineClient, err := client.NewPythonClient("localhost:50051")
	if err != nil {
		log.Fatalf("failed to connect to engine: %v", err)
	}
	defer engineClient.Close()

	promptHandler := handler.NewPromptHandler(engineClient)
	mux := http.NewServeMux()
	mux.Handle("/prompt", middleware.Audit(promptHandler))

	log.Println("Imprimer gateway listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", mux))
}
