package ui

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
)

// Banner renders the Imprimer identity header.
// Call once at the start of any command output.
func Banner() string {
	art := strings.Join([]string{
		" ██╗███╗   ███╗██████╗ ██████╗ ██╗███╗   ███╗███████╗██████╗ ",
		" ██║████╗ ████║██╔══██╗██╔══██╗██║████╗ ████║██╔════╝██╔══██╗",
		" ██║██╔████╔██║██████╔╝██████╔╝██║██╔████╔██║█████╗  ██████╔╝",
		" ██║██║╚██╔╝██║██╔═══╝ ██╔══██╗██║██║╚██╔╝██║██╔══╝  ██╔══██╗",
		" ██║██║ ╚═╝ ██║██║     ██║  ██║██║██║ ╚═╝ ██║███████╗██║  ██║",
		" ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝",
	}, "\n")

	tagline := DimStyle.Render("  prompt control · reachability · optimization")

	return lipgloss.NewStyle().
		Foreground(NeonBlue).
		MarginBottom(1).
		Render(art) + "\n" + tagline + "\n"
}

// Metric renders a single label/value pair on one line.
func Metric(label string, value any) string {
	return fmt.Sprintf(
		"%s %s",
		LabelStyle.Render(fmt.Sprintf("%-20s", label)),
		ValueStyle.Render(fmt.Sprint(value)),
	)
}

// SectionTitle renders a section header inside a panel.
func SectionTitle(title string) string {
	return SectionStyle.Render(title)
}

// Prompt renders prompt text in the prompt style.
func Prompt(text string) string {
	return PromptStyle.Render(text)
}

// Panel wraps content in a rounded border with a title.
func Panel(title string, body string) string {
	content := TitleStyle.Render(title) + "\n\n" + strings.TrimSpace(body)
	return PanelStyle.Render(content)
}

// WinnerPanel is Panel with an amber border — used for the winning variant.
func WinnerPanel(title string, body string) string {
	content := WinnerStyle.Render(title) + "\n\n" + strings.TrimSpace(body)
	return WinnerPanelStyle.Render(content)
}

// Table renders a two-column comparison table.
// headers: [left header, right header]
// rows: each row is [left cell, right cell]
func Table(headers [2]string, rows [][2]string) string {
	colWidth := 36

	header := fmt.Sprintf(
		"%s  %s",
		TableHeaderStyle.Width(colWidth).Render(headers[0]),
		TableHeaderStyle.Width(colWidth).Render(headers[1]),
	)

	var lines []string
	lines = append(lines, header)

	for _, row := range rows {
		line := fmt.Sprintf(
			"%s  %s",
			lipgloss.NewStyle().Width(colWidth).Render(row[0]),
			lipgloss.NewStyle().Width(colWidth).Render(row[1]),
		)
		lines = append(lines, line)
	}

	return strings.Join(lines, "\n")
}

// ScoreBar renders a visual bar proportional to a 0.0-1.0 score.
// width is in characters.
func ScoreBar(score float32, width int) string {
	filled := int(score * float32(width))
	if filled > width {
		filled = width
	}
	empty := width - filled

	bar := strings.Repeat("█", filled) + strings.Repeat("░", empty)

	color := NeonPink
	switch {
	case score >= 0.75:
		color = NeonGreen
	case score >= 0.50:
		color = NeonAmber
	default:
		color = NeonPink
	}

	return lipgloss.NewStyle().Foreground(color).Render(bar)
}
