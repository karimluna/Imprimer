package httpx

import (
	"encoding/json"
	"net/http"
)

// Centralized HTTP helpers

func DecodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	// Ensure body is always closed to avoid resource leaks
	defer r.Body.Close()

	// Create decoder for request body
	decoder := json.NewDecoder(r.Body)

	// Reject unknown fieldss
	decoder.DisallowUnknownFields()

	// Decode JSON into target struct
	if err := decoder.Decode(v); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return false
	}

	return true
}

func WriteJSON(w http.ResponseWriter, status int, v any) {
	// Enforce JSON responses across the API
	w.Header().Set("Content-Type", "application/json")

	// Set HTTP status code before writing body
	w.WriteHeader(status)

	// Encode response struct
	if err := json.NewEncoder(w).Encode(v); err != nil {
		// Falkback error if encoding fails (rare but important)
		http.Error(w, "failed to encode response", http.StatusInternalServerError)
	}
}

func WriteError(w http.ResponseWriter, status int, msg string) {
	WriteJSON(w, status, map[string]any{
		"error": msg,
	})
}
