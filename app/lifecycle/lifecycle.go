package lifecycle

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tray"
)

func Run() {
	InitLogging()

	ctx, cancel := context.WithCancel(context.Background())
	var done chan int

	t, err := tray.NewTray()
	if err != nil {
		log.Fatalf("Failed to start: %s", err)
	}
	callbacks := t.GetCallbacks()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)

	// TODO - maybe move elsewhere
	if store.GetAllowExternalConnections() && os.Getenv("OLLAMA_HOST") == "" {
		os.Setenv("OLLAMA_HOST", "0.0.0.0")
	}
	if store.GetAllowBrowserConnections() && os.Getenv("OLLAMA_ORIGINS") == "" {
		os.Setenv("OLLAMA_ORIGINS", "*")
	}
	modelDir := store.GetModelDir()
	if modelDir != "" && os.Getenv("OLLAMA_MODELS") == "" {
		os.Setenv("OLLAMA_MODELS", modelDir)

	}

	restartServer := func() {
		cancel()
		slog.Info("Waiting for ollama server to shutdown...")
		if done != nil {
			<-done
		}
		slog.Info("Restarting ollama server with new settings...")
		ctx, cancel = context.WithCancel(context.Background())
		done, err = SpawnServer(ctx, CLIName)
		if err != nil {
			// TODO - should we retry in a backoff loop?
			// TODO - should we pop up a warning and maybe add a menu item to view application logs?
			slog.Error(fmt.Sprintf("Failed to spawn ollama server %s", err))
			done = make(chan int, 1)
			done <- 1
		}

	}

	go func() {
		slog.Debug("starting callback loop")
		for {
			select {
			case <-callbacks.Quit:
				slog.Debug("quit called")
				t.Quit()
			case <-signals:
				slog.Debug("shutting down due to signal")
				t.Quit()
			case <-callbacks.Update:
				err := DoUpgrade(cancel, done)
				if err != nil {
					slog.Warn(fmt.Sprintf("upgrade attempt failed: %s", err))
				}
			case <-callbacks.ShowLogs:
				ShowLogs()
			case <-callbacks.DoFirstUse:
				err := GetStarted()
				if err != nil {
					slog.Warn(fmt.Sprintf("Failed to launch getting started shell: %s", err))
				}
			case val := <-callbacks.ExposeHost:
				if val {
					os.Setenv("OLLAMA_HOST", "0.0.0.0")
				} else {
					os.Setenv("OLLAMA_HOST", "")
				}
				restartServer()
			case val := <-callbacks.ExposeBrowser:
				if val {
					os.Setenv("OLLAMA_ORIGINS", "*")
				} else {
					os.Setenv("OLLAMA_ORIGINS", "")
				}
				restartServer()
			case val := <-callbacks.UpdateModelDir:
				// TODO - should we validate the path?  What if it fails?
				os.Setenv("OLLAMA_MODELS", val)
				restartServer()
			}
		}
	}()

	// Are we first use?
	if !store.GetFirstTimeRun() {
		slog.Debug("First time run")
		err = t.DisplayFirstUseNotification()
		if err != nil {
			slog.Debug(fmt.Sprintf("XXX failed to display first use notification %v", err))
		}
		store.SetFirstTimeRun(true)
	} else {
		slog.Debug("Not first time, skipping first run notification")
	}

	if IsServerRunning(ctx) {
		slog.Info("Detected another instance of ollama running, exiting")
		os.Exit(1)
	} else {
		done, err = SpawnServer(ctx, CLIName)
		if err != nil {
			// TODO - should we retry in a backoff loop?
			// TODO - should we pop up a warning and maybe add a menu item to view application logs?
			slog.Error(fmt.Sprintf("Failed to spawn ollama server %s", err))
			done = make(chan int, 1)
			done <- 1
		}
	}

	StartBackgroundUpdaterChecker(ctx, t.UpdateAvailable)

	t.Run()
	cancel()
	slog.Info("Waiting for ollama server to shutdown...")
	if done != nil {
		<-done
	}
	slog.Info("Ollama app exiting")
}
