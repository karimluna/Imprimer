package ui

import "charm.land/lipgloss/v2"

var (
	NeonBlue  = lipgloss.Color("#00D9FF")
	NeonPink  = lipgloss.Color("#FF2AF0")
	NeonGreen = lipgloss.Color("#00FF9C")
	NeonAmber = lipgloss.Color("#FFB347")
	Muted     = lipgloss.Color("#7D7D91")
	Subtle    = lipgloss.Color("#3D3D4D")

	TitleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(NeonBlue)

	LabelStyle = lipgloss.NewStyle().
			Foreground(Muted)

	ValueStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(NeonGreen)

	WinnerStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(NeonAmber)

	DimStyle = lipgloss.NewStyle().
			Foreground(Subtle)

	PromptStyle = lipgloss.NewStyle().
			Foreground(NeonPink).
			Italic(true)

	SectionStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(NeonBlue)

	PanelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(NeonBlue).
			Padding(1, 2).
			MarginTop(1)

	WinnerPanelStyle = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(NeonAmber).
				Padding(1, 2).
				MarginTop(1)

	TableHeaderStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(NeonBlue).
				BorderBottom(true).
				BorderStyle(lipgloss.NormalBorder()).
				BorderForeground(Subtle)
)
