package handler

import (
	"encoding/json"
	"net/http"
	"strconv"

	gen "github.com/BalorLC3/Imprimer/gateway/gen"
	"github.com/BalorLC3/Imprimer/gateway/internal/client"
)

type BestHandler struct {
	engine *client.PythonClient
}

func NewBestHandler(engine *client.PythonClient) *BestHandler {
	return &BestHandler{engine: engine}
}

func (h *BestHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	task := r.URL.Query().Get("task")
	if task == "" {
		http.Error(w, "task query parameter required", http.StatusBadRequest)
		return
	}

	limit := 10
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			limit = parsed
		}
	}

	resp, err := h.engine.Best(r.Context(), &gen.BestRequest{
		Task:  task,
		Limit: int32(limit),
	})
	if err != nil {
		http.Error(w, "engine error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if !resp.Found {
		http.Error(w, "no evaluations found for task: "+task, http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"task":             resp.Task,
		"best_template":    resp.BestTemplate,
		"avg_reachability": resp.AvgReachability,
		"avg_score":        resp.AvgScore,
		"evaluations":      resp.Evaluations,
	})
}
