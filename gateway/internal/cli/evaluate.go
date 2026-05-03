package cli

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/BalorLC3/imprimer/gateway/internal/ui"
	"github.com/spf13/cobra"
)

type evaluatePayload struct {
	Task     string `json:"task"`
	Input    string `json:"input"`
	VariantA string `json:"variant_a"`
	VariantB string `json:"variant_b"`
	Backend  string `json:"backend"`
}

type evaluateResult struct {
	TraceID  string  `json:"trace_id"`
	Winner   string  `json:"winner"`
	OutputA  string  `json:"output_a"`
	OutputB  string  `json:"output_b"`
	LatencyA float32 `json:"latency_a_ms"`
	LatencyB float32 `json:"latency_b_ms"`
	ScoreA   float32 `json:"score_a"`
	ScoreB   float32 `json:"score_b"`
}

var evaluateCmd = &cobra.Command{
	Use:   "evaluate",
	Short: "Run two prompt variants and compare their reachability scores",
	Example: `  imprimer evaluate \
    --task summarize \
    --input "Minsky argued intelligence emerges from many small agents" \
    --a "Summarize this in one sentence: {input}" \
    --b "You are an expert writer. Give a precise one sentence summary of: {input}"`,
	RunE: func(cmd *cobra.Command, args []string) error {
		task, _ := cmd.Flags().GetString("task")
		input, _ := cmd.Flags().GetString("input")
		variantA, _ := cmd.Flags().GetString("a")
		variantB, _ := cmd.Flags().GetString("b")
		backend, _ := cmd.Flags().GetString("backend")

		if task == "" || variantA == "" || variantB == "" {
			return fmt.Errorf("--task, --a, and --b are required")
		}

		c := NewImprimerClient(gatewayURL, apiKey)

		var result evaluateResult
		if err := c.post("/prompt", evaluatePayload{
			Task:     task,
			Input:    input,
			VariantA: variantA,
			VariantB: variantB,
			Backend:  backend,
		}, &result); err != nil {
			return err
		}

		if outputJSON {
			raw, _ := json.MarshalIndent(result, "", "  ")
			fmt.Println(string(raw))
			return nil
		}

		fmt.Println(ui.Banner())

		summary := strings.Join([]string{
			ui.Metric("Trace ID", result.TraceID),
			ui.Metric("Task", task),
			ui.Metric("Backend", backend),
			ui.Metric("Winner", "Variant "+strings.ToUpper(result.Winner)),
		}, "\n")

		fmt.Println(ui.Panel("Evaluation", summary))

		barA := ui.ScoreBar(result.ScoreA, 20)
		barB := ui.ScoreBar(result.ScoreB, 20)

		table := ui.Table(
			[2]string{"Variant A", "Variant B"},
			[][2]string{
				{
					fmt.Sprintf("Score    %.3f", result.ScoreA),
					fmt.Sprintf("Score    %.3f", result.ScoreB),
				},
				{
					fmt.Sprintf("Latency  %.0fms", result.LatencyA),
					fmt.Sprintf("Latency  %.0fms", result.LatencyB),
				},
				{barA, barB},
			},
		)

		fmt.Println(ui.Panel("Scores", table))

		// Loser gets a regular panel, winner gets the amber panel
		buildBody := func(prompt, output string) string {
			return ui.Prompt(truncate(prompt, 80)) +
				"\n\n" +
				output
		}

		if result.Winner == "a" {
			fmt.Println(ui.WinnerPanel("Variant A  ★", buildBody(variantA, result.OutputA)))
			fmt.Println(ui.Panel("Variant B", buildBody(variantB, result.OutputB)))
		} else {
			fmt.Println(ui.Panel("Variant A", buildBody(variantA, result.OutputA)))
			fmt.Println(ui.WinnerPanel("Variant B  ★", buildBody(variantB, result.OutputB)))
		}

		return nil
	},
}

func init() {
	evaluateCmd.Flags().String("task", "", "Task type (summarize, classify, extract)")
	evaluateCmd.Flags().String("input", "", "Input text (optional for some tasks)")
	evaluateCmd.Flags().String("a", "", "First prompt template")
	evaluateCmd.Flags().String("b", "", "Second prompt template")
	evaluateCmd.Flags().String("backend", "ollama", "ollama or openai")

	RootCmd.AddCommand(evaluateCmd)
}
