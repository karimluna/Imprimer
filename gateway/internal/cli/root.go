package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// Global flags available to all subcommands

var (
	gatewayURL string
	apiKey     string
	outputJSON bool
)

// Inspired by the guides in https://github.com/spf13/cobra?tab=readme-ov-file
var RootCmd = &cobra.Command{
	Use:   "imprimer",
	Short: "Imprimer - LLM prompt control platform CLI",
	Long: `Imprimer measures and optimizes how precisely a prompt
controls an LLM's output distribution.

Grounded on "What's the Magic Word? A Control Theory of LLM Prompting"
(Bhargava et al., 2023) and Minsky's Society of Mind.

Examples:
  imprimer evaluate --task summarize --input "your text" --a "prompt A" --b "prompt B"
  imprimer optimize --task summarize --trials 20
  imprimer best --task summarize`,
}

func init() {
	RootCmd.PersistentFlags().StringVar(
		&gatewayURL, "gateway", "http://localhost:8080",
		"Imprimer gateway URL",
	)
	RootCmd.PersistentFlags().StringVar(
		&apiKey, "api-key", os.Getenv("IMPRIMER_API_KEY"),
		"API key (defaults to IMPRIMER_API_KEY env var)",
	)
	RootCmd.PersistentFlags().BoolVar(
		&outputJSON, "json", false,
		"Output raw JSON instead of formatted text",
	)
}

// Execute is called by main — runs the root command and exits on error
func Execute() {
	if err := RootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
