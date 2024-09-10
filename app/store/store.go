package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/uuid"
)

type Store struct {
	ID                       string `json:"id"`
	FirstTimeRun             bool   `json:"first-time-run"`
	AllowExternalConnections bool   `json:"external-connections"`
	AllowBrowserConnections  bool   `json:"browser-connections"`
	ModelDir                 string `json:"model-dir"`
}

var (
	lock  sync.Mutex
	store Store
)

func GetID() string {
	lock.Lock()
	defer lock.Unlock()
	if store.ID == "" {
		initStore()
	}
	return store.ID
}

func GetFirstTimeRun() bool {
	lock.Lock()
	defer lock.Unlock()
	if store.ID == "" {
		initStore()
	}
	return store.FirstTimeRun
}

func SetFirstTimeRun(val bool) {
	lock.Lock()
	defer lock.Unlock()
	if store.FirstTimeRun == val {
		return
	}
	store.FirstTimeRun = val
	writeStore(getStorePath())
}

func GetAllowExternalConnections() bool {
	lock.Lock()
	defer lock.Unlock()
	if store.ID == "" {
		initStore()
	}
	return store.AllowExternalConnections
}

func SetAllowExternalConnections(val bool) {
	lock.Lock()
	defer lock.Unlock()
	if store.AllowExternalConnections == val {
		return
	}
	store.AllowExternalConnections = val
	writeStore(getStorePath())
}

func GetAllowBrowserConnections() bool {
	lock.Lock()
	defer lock.Unlock()
	if store.ID == "" {
		initStore()
	}
	return store.AllowBrowserConnections
}

func SetAllowBrowserConnections(val bool) {
	lock.Lock()
	defer lock.Unlock()
	if store.AllowBrowserConnections == val {
		return
	}
	store.AllowBrowserConnections = val
	writeStore(getStorePath())
}

func GetModelDir() string {
	lock.Lock()
	defer lock.Unlock()
	if store.ID == "" {
		initStore()
	}
	return store.ModelDir
}

func SetModelDir(val string) {
	lock.Lock()
	defer lock.Unlock()
	if store.ModelDir == val {
		return
	}
	store.ModelDir = val
	writeStore(getStorePath())
}

// lock must be held
func initStore() {
	storeFile, err := os.Open(getStorePath())
	if err == nil {
		defer storeFile.Close()
		err = json.NewDecoder(storeFile).Decode(&store)
		if err == nil {
			slog.Debug(fmt.Sprintf("loaded existing store %s - ID: %s", getStorePath(), store.ID))
			return
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		slog.Debug(fmt.Sprintf("unexpected error searching for store: %s", err))
	}
	slog.Debug("initializing new store")
	store.ID = uuid.New().String()
	writeStore(getStorePath())
}

func writeStore(storeFilename string) {
	ollamaDir := filepath.Dir(storeFilename)
	_, err := os.Stat(ollamaDir)
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(ollamaDir, 0o755); err != nil {
			slog.Error(fmt.Sprintf("create ollama dir %s: %v", ollamaDir, err))
			return
		}
	}
	payload, err := json.Marshal(store)
	if err != nil {
		slog.Error(fmt.Sprintf("failed to marshal store: %s", err))
		return
	}
	fp, err := os.OpenFile(storeFilename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		slog.Error(fmt.Sprintf("write store payload %s: %v", storeFilename, err))
		return
	}
	defer fp.Close()
	if n, err := fp.Write(payload); err != nil || n != len(payload) {
		slog.Error(fmt.Sprintf("write store payload %s: %d vs %d -- %v", storeFilename, n, len(payload), err))
		return
	}
	slog.Debug("Store contents: " + string(payload))
	slog.Info(fmt.Sprintf("wrote store: %s", storeFilename))
}
