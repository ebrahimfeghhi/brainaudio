# Important Commands

## vLLM Server

### Start vLLM server with Gemma 4 31B IT
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-31B-it --enable-auto-tool-choice --tool-call-parser pythonic --max-model-len 131072
```
Starts a local vLLM inference server on `http://localhost:8000/v1` using Gemma 4 31B IT on GPU 0. The `--enable-auto-tool-choice` and `--tool-call-parser pythonic` flags are required for function/tool calling support with Gemma models.


```bash
tmux new -s vllm
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-31B-it --enable-auto-tool-choice --tool-call-parser pythonic --max-model-len 131072

