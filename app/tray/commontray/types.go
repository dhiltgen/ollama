package commontray

var (
	Title   = "Ollama"
	ToolTip = "Ollama"

	UpdateIconName = "tray_upgrade"
	IconName       = "tray"
)

type Callbacks struct {
	Quit           chan struct{}
	Update         chan struct{}
	DoFirstUse     chan struct{}
	ShowLogs       chan struct{}
	ExposeHost     chan bool
	ExposeBrowser  chan bool
	UpdateModelDir chan string
}

type OllamaTray interface {
	GetCallbacks() Callbacks
	Run()
	UpdateAvailable(ver string) error
	DisplayFirstUseNotification() error
	Quit()
}
