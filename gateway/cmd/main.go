package main

import (
	"log"
	"net/http"

	"github.com/BalorLC3/imprimer/gateway/internal/client"
	"github.com/BalorLC3/imprimer/gateway/internal/handler"
	"github.com/BalorLC3/imprimer/gateway/internal/middleware"
)

func main() {
	// Connect to Python gRPC server
	engineClient, err := client.NewPythonClient("localhost:50051")
	if err != nil {
		log.Fatalf("failed to connect to engine: %v", err)
	}
	defer engineClient.Close()

	promptHandler := handler.NewPromptHandler(engineClient)
	mux := http.NewServerMux()
	mux.Handle("/prompt", middleware.Audit(promptHandler))

	log.Prin

}
