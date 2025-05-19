package ADLX

import "testing"

func TestAMD(t *testing.T) {
	_, err := GetGPUs()
	if err != nil {
		t.Fatal(err)
	}
}
