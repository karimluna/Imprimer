package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Imprimer is a thin HTTP client that talks to the gateway.
// All CLI commands share this client, it's the only place
// that knows the gateway URL and API key.

type ImprimerClient struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

func NewImprimerClient(baseURL, apiKey string) *ImprimerClient {
	return &ImprimerClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 10 * time.Minute,
		},
	}
}

func (c *ImprimerClient) post(path string, body any, out any) error {
	payload, err := json.Marshal(body) // convert struct to json-encoded byte slice
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, c.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	req.Header.Set("Content-type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed : %w", err)
	}
	defer resp.Body.Close() // remember LIFO

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return fmt.Errorf("gateway returned %d: %s", resp.StatusCode, string(raw))
	}

	return json.Unmarshal(raw, out)
}

func (c *ImprimerClient) get(path string, out any) error {
	req, err := http.NewRequest(http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return fmt.Errorf("gateway returned %d: %s", resp.StatusCode, string(raw))
	}

	return json.Unmarshal(raw, out)
}
