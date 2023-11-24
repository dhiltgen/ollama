package server

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/jmorganca/ollama/api"
)

// TODO - this would ideally be in the llm package, but that would require some refactoring of interfaces in the server
//        package to avoid circular dependencies

var (
	req = [2]api.GenerateRequest{
		{
			Model:   "orca-mini",
			Prompt:  "tell me a short story about agi?",
			Options: map[string]interface{}{},
		}, {
			Model:   "orca-mini",
			Prompt:  "what is the origin of the us thanksgiving holiday?",
			Options: map[string]interface{}{},
		},
	}
	resp = [2]string{
		"once upon a time",
		"english colonists who",
	}
)

func TestIntegrationSimpleOrcaMini(t *testing.T) {
	SkipIFNoTestData(t)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*60)
	defer cancel()
	opts := api.DefaultOptions()
	opts.Temperature = 0.0
	model, llmRunner := PrepareModelForPrompts(t, req[0].Model, opts)
	defer llmRunner.Close()
	response := OneShotPromptResponse(t, ctx, req[0], model, llmRunner)
	assert.Contains(t, strings.ToLower(response), resp[0])
}

// TODO
// The server always loads a new runner and closes the old one, which forces serial execution
// At present this test case fails with concurrency problems.  Eventually we should try to
// get true concurrency working with n_parallel support in the backend
func TestIntegrationConcurrentPredictOrcaMini(t *testing.T) {
	SkipIFNoTestData(t)
	t.Skip("concurrent prediction on single runner not currently supported")
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*60)
	defer cancel()
	opts := api.DefaultOptions()
	opts.Temperature = 0.0
	var wg sync.WaitGroup
	wg.Add(len(req))
	model, llmRunner := PrepareModelForPrompts(t, req[0].Model, opts)
	defer llmRunner.Close()
	for i := 0; i < len(req); i++ {
		go func(i int) {
			defer wg.Done()
			response := OneShotPromptResponse(t, ctx, req[i], model, llmRunner)
			t.Logf("Prompt: %s\nResponse: %s", req[0].Prompt, response)
			assert.Contains(t, strings.ToLower(response), resp[i], "error in thread %d (%s)", i, req[i].Prompt)
		}(i)
	}
	wg.Wait()
}

func TestIntegrationConcurrentRunnersOrcaMini(t *testing.T) {
	SkipIFNoTestData(t)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*60)
	defer cancel()
	opts := api.DefaultOptions()
	opts.Temperature = 0.0
	var wg sync.WaitGroup
	wg.Add(len(req))

	for i := 0; i < len(req); i++ {
		go func(i int) {
			defer wg.Done()
			model, llmRunner := PrepareModelForPrompts(t, req[0].Model, opts)
			defer llmRunner.Close()
			response := OneShotPromptResponse(t, ctx, req[i], model, llmRunner)
			t.Logf("Prompt: %s\nResponse: %s", req[0].Prompt, response)
			assert.Contains(t, strings.ToLower(response), resp[i], "error in thread %d (%s)", i, req[i].Prompt)
		}(i)
	}
	wg.Wait()
}

// TODO - create a parallel test with 2 different models once we support concurrency
