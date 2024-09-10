//go:build windows

package commontray

const (
	FirstTimeTitle   = "Ollama is running"
	FirstTimeMessage = "Click here to get started"
	UpdateTitle      = "Update available"
	UpdateMessage    = "Ollama version %s is ready to install"

	QuitMenuTitle            = "Quit Ollama"
	UpdateAvailableMenuTitle = "An update is available"
	UpdateMenutTitle         = "Restart to update"
	DiagLogsMenuTitle        = "View logs"
	SettingsMenuTitle        = "Settings"
	HostMenuTitle            = "Allow external connections"
	DomainMenuTitle          = "Allow browser connections"
	MenuDirTitle             = "Choose model directory"

	ModelDialogMessage = "Please pick the Ollama Model directory\r\r(Any existing models will not be moved.)"
)
