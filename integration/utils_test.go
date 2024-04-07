//go:build integration

package integration

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/lifecycle"
	"github.com/stretchr/testify/assert"
)

func FindPort() string {
	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		var l *net.TCPListener
		if l, err = net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		port = rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
	}
	return strconv.Itoa(port)
}

func GetTestEndpoint() (*api.Client, string) {
	defaultPort := "11434"
	ollamaHost := os.Getenv("OLLAMA_HOST")

	scheme, hostport, ok := strings.Cut(ollamaHost, "://")
	if !ok {
		scheme, hostport = "http", ollamaHost
	}

	// trim trailing slashes
	hostport = strings.TrimRight(hostport, "/")

	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "127.0.0.1", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if os.Getenv("OLLAMA_TEST_EXISTING") == "" && port == defaultPort {
		port = FindPort()
	}

	slog.Info("server connection", "host", host, "port", port)

	return api.NewClient(
		&url.URL{
			Scheme: scheme,
			Host:   net.JoinHostPort(host, port),
		},
		http.DefaultClient), fmt.Sprintf("%s:%s", host, port)
}

var serverMutex sync.Mutex
var serverReady bool

func StartServer(ctx context.Context, ollamaHost string) error {
	// Make sure the server has been built
	CLIName, err := filepath.Abs("../ollama")
	if err != nil {
		return err
	}

	if runtime.GOOS == "windows" {
		CLIName += ".exe"
	}
	_, err = os.Stat(CLIName)
	if err != nil {
		return fmt.Errorf("CLI missing, did you forget to build first?  %w", err)
	}
	serverMutex.Lock()
	defer serverMutex.Unlock()
	if serverReady {
		return nil
	}

	if tmp := os.Getenv("OLLAMA_HOST"); tmp != ollamaHost {
		slog.Info("setting env", "OLLAMA_HOST", ollamaHost)
		os.Setenv("OLLAMA_HOST", ollamaHost)
	}

	slog.Info("starting server", "url", ollamaHost)
	done, err := lifecycle.SpawnServer(ctx, "../ollama")
	if err != nil {
		return fmt.Errorf("failed to start server: %w", err)
	}

	go func() {
		<-ctx.Done()
		serverMutex.Lock()
		defer serverMutex.Unlock()
		exitCode := <-done
		if exitCode > 0 {
			slog.Warn("server failure", "exit", exitCode)
		}
		serverReady = false
	}()

	// TODO wait only long enough for the server to be responsive...
	time.Sleep(500 * time.Millisecond)

	serverReady = true
	return nil
}

func PullIfMissing(ctx context.Context, client *api.Client, modelName string) error {
	slog.Info("checking status of model", "model", modelName)
	showReq := &api.ShowRequest{Name: modelName}

	showCtx, cancel := context.WithDeadlineCause(
		ctx,
		time.Now().Add(5*time.Second),
		fmt.Errorf("show for existing model %s took too long", modelName),
	)
	defer cancel()
	_, err := client.Show(showCtx, showReq)
	var statusError api.StatusError
	switch {
	case errors.As(err, &statusError) && statusError.StatusCode == http.StatusNotFound:
		break
	case err != nil:
		return err
	default:
		slog.Info("model already present", "model", modelName)
		return nil
	}
	slog.Info("model missing", "model", modelName)

	stallDuration := 5 * time.Second
	stallTimer := time.NewTimer(stallDuration)
	fn := func(resp api.ProgressResponse) error {
		// fmt.Print(".")
		if !stallTimer.Reset(stallDuration) {
			return fmt.Errorf("stall was detected, aborting status reporting")
		}
		return nil
	}

	stream := true
	pullReq := &api.PullRequest{Name: modelName, Stream: &stream}

	var pullError error

	done := make(chan int)
	go func() {
		pullError = client.Pull(ctx, pullReq, fn)
		done <- 0
	}()

	select {
	case <-stallTimer.C:
		return fmt.Errorf("download stalled")
	case <-done:
		return pullError
	}
}

var serverProcMutex sync.Mutex

func GenerateTestHelper(ctx context.Context, t *testing.T, genReq api.GenerateRequest, anyResp []string) {

	// TODO maybe stuff in an init routine?
	lifecycle.InitLogging()

	defer func() {
		if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
			defer serverProcMutex.Unlock()
			if t.Failed() {
				fp, err := os.Open(lifecycle.ServerLogFile)
				if err != nil {
					slog.Error("failed to open server log", "logfile", lifecycle.ServerLogFile, "error", err)
					return
				}
				data, err := io.ReadAll(fp)
				if err != nil {
					slog.Error("failed to read server log", "logfile", lifecycle.ServerLogFile, "error", err)
					return
				}
				slog.Warn("SERVER LOG FOLLOWS")
				os.Stderr.Write(data)
				slog.Warn("END OF SERVER")
			}
			err := os.Remove(lifecycle.ServerLogFile)
			if err != nil && !os.IsNotExist(err) {
				slog.Warn("failed to cleanup", "logfile", lifecycle.ServerLogFile, "error", err)
			}
		}
	}()
	client, testEndpoint := GetTestEndpoint()

	if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
		serverProcMutex.Lock()
		fp, err := os.CreateTemp("", "ollama-server-*.log")
		if err != nil {
			t.Fatalf("failed to generate log file: %s", err)
		}
		lifecycle.ServerLogFile = fp.Name()
		fp.Close()
		assert.NoError(t, StartServer(ctx, testEndpoint))
	}

	assert.NoError(t, PullIfMissing(ctx, client, genReq.Model))

	stallTimer := time.NewTimer(30 * time.Second) // Initial model load can take some time...
	var buf bytes.Buffer
	fn := func(response api.GenerateResponse) error {
		// fmt.Print(".")
		buf.Write([]byte(response.Response))
		if !stallTimer.Reset(10 * time.Second) {
			return fmt.Errorf("stall was detected while streaming response, aborting")
		}
		return nil
	}

	stream := true
	genReq.Stream = &stream
	done := make(chan int)
	var genErr error
	go func() {
		genErr = client.Generate(ctx, &genReq, fn)
		done <- 0
	}()

	select {
	case <-stallTimer.C:
		t.Errorf("generate stalled.  Response so far:%s", buf.String())
	case <-done:
		assert.NoError(t, genErr)
		// Verify the response contains the expected data
		response := buf.String()
		atLeastOne := false
		for _, resp := range anyResp {
			if strings.Contains(strings.ToLower(response), resp) {
				atLeastOne = true
				break
			}
		}
		assert.True(t, atLeastOne, "none of %v found in %s", anyResp, response)
		slog.Info("test pass", "prompt", genReq.Prompt, "contains", anyResp, "response", response)
	case <-ctx.Done():
		t.Error("outer test context done while waiting for generate")
	}
}
