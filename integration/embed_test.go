//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestAllMiniLMEmbeddings(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbeddingRequest{
		Model:  "all-minilm",
		Prompt: "why is the sky blue?",
	}

	res, err := embeddingTestHelper(ctx, t, req)

	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if len(res.Embedding) != 384 {
		t.Fatalf("expected 384 floats, got %d", len(res.Embedding))
	}

	// Different GPU kernels have slightly different behavior
	min := float64(0.064948)
	max := float64(0.068516)
	if res.Embedding[0] < min || res.Embedding[0] > max {
		t.Fatalf("expected in the range %.16f - %.16f, got %.16f", min, max, res.Embedding[0])
	}
}

func TestAllMiniLMEmbed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbedRequest{
		Model: "all-minilm",
		Input: "why is the sky blue?",
	}

	res, err := embedTestHelper(ctx, t, req)

	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if len(res.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(res.Embeddings))
	}

	if len(res.Embeddings[0]) != 384 {
		t.Fatalf("expected 384 floats, got %d", len(res.Embeddings[0]))
	}

	// Different GPU kernels have slightly different behavior
	min := float32(0.00006309)
	max := float32(0.01038676)
	if res.Embeddings[0][0] < min || res.Embeddings[0][0] > max {
		t.Fatalf("expected in the range %.8f - %.8f, got %.8f", min, max, res.Embeddings[0][0])
	}

	if res.PromptEvalCount != 6 {
		t.Fatalf("expected 6 prompt tokens, got %d", res.PromptEvalCount)
	}
}

func TestAllMiniLMBatchEmbed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbedRequest{
		Model: "all-minilm",
		Input: []string{"why is the sky blue?", "why is the grass green?"},
	}

	res, err := embedTestHelper(ctx, t, req)

	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if len(res.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(res.Embeddings))
	}

	if len(res.Embeddings[0]) != 384 {
		t.Fatalf("expected 384 floats, got %d", len(res.Embeddings[0]))
	}

	// Different GPU kernels have slightly different behavior
	min0 := float32(0.00984993)
	max0 := float32(0.03093774)
	min1 := float32(-0.04268764)
	max1 := float32(-0.00977226)
	if res.Embeddings[0][0] < min0 || res.Embeddings[0][0] > max0 || res.Embeddings[1][0] < min1 || res.Embeddings[1][0] > max1 {
		t.Fatalf("expected between %.8f - %.8f and %.8f - %.8f, got %.8f and %.8f", min0, max0, min1, max1, res.Embeddings[0][0], res.Embeddings[1][0])
	}

	if res.PromptEvalCount != 12 {
		t.Fatalf("expected 12 prompt tokens, got %d", res.PromptEvalCount)
	}
}

func TestAllMiniLMEmbedTruncate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	truncTrue, truncFalse := true, false

	type testReq struct {
		Name    string
		Request api.EmbedRequest
	}

	reqs := []testReq{
		{
			Name: "Target Truncation",
			Request: api.EmbedRequest{
				Model: "all-minilm",
				Input: "why",
			},
		},
		{
			Name: "Default Truncate",
			Request: api.EmbedRequest{
				Model:   "all-minilm",
				Input:   "why is the sky blue?",
				Options: map[string]any{"num_ctx": 1},
			},
		},
		{
			Name: "Explicit Truncate",
			Request: api.EmbedRequest{
				Model:    "all-minilm",
				Input:    "why is the sky blue?",
				Truncate: &truncTrue,
				Options:  map[string]any{"num_ctx": 1},
			},
		},
	}

	res := make(map[string]*api.EmbedResponse)

	for _, req := range reqs {
		response, err := embedTestHelper(ctx, t, req.Request)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		res[req.Name] = response
	}

	if res["Target Truncation"].Embeddings[0][0] != res["Default Truncate"].Embeddings[0][0] {
		t.Fatal("expected default request to truncate correctly")
	}

	if res["Default Truncate"].Embeddings[0][0] != res["Explicit Truncate"].Embeddings[0][0] {
		t.Fatal("expected default request and truncate true request to be the same")
	}

	// check that truncate set to false returns an error if context length is exceeded
	_, err := embedTestHelper(ctx, t, api.EmbedRequest{
		Model:    "all-minilm",
		Input:    "why is the sky blue?",
		Truncate: &truncFalse,
		Options:  map[string]any{"num_ctx": 1},
	})

	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func embeddingTestHelper(ctx context.Context, t *testing.T, req api.EmbeddingRequest) (*api.EmbeddingResponse, error) {
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("failed to pull model %s: %v", req.Model, err)
	}

	response, err := client.Embeddings(ctx, &req)

	if err != nil {
		return nil, err
	}

	return response, nil
}

func embedTestHelper(ctx context.Context, t *testing.T, req api.EmbedRequest) (*api.EmbedResponse, error) {
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("failed to pull model %s: %v", req.Model, err)
	}

	response, err := client.Embed(ctx, &req)

	if err != nil {
		return nil, err
	}

	return response, nil
}
