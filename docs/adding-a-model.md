# Adding a New Model

How to add a model to one or more nodes in the fleet and keep config in sync.

---

## Step 1 — Check if the model is on the Ollama registry

```bash
ollama search <model-name>
```

Or check [ollama.com/library](https://ollama.com/library). If it's there, use **Path A**. If not, use **Path B**.

---

## Path A — Ollama-native model

### On the target node

```bash
ssh -i ~/.ssh/id_ed25519 wally@192.168.1.52   # lenovo-gb10
# or johanus@192.168.1.211                     # nvidia-ai

ollama pull <tag>           # e.g. ollama pull qwen3.5:35b-a3b
ollama list                 # confirm it landed
```

### In the setup script

Add to the relevant `ollama_pull` section in `scripts/setup/setup-gb10.sh` (or `setup-nvidia.sh` / `setup-strix.sh`):

```bash
ollama_pull <tag>
```

---

## Path B — HuggingFace GGUF (not on Ollama registry)

### Find the right file

1. Search [huggingface.co/bartowski](https://huggingface.co/bartowski) or similar quantization repos
2. Pick `Q4_K_M` quant — good balance of quality and size
3. Note the **repo** (`bartowski/Qwen_Qwen3.5-27B-GGUF`) and **filename** (`Qwen_Qwen3.5-27B-Q4_K_M.gguf`)

### Check available filenames in a repo

```bash
curl -s "https://huggingface.co/api/models/<owner>/<repo>" \
  | python3 -c 'import sys,json; [print(s["rfilename"]) for s in json.load(sys.stdin)["siblings"] if "Q4_K_M" in s["rfilename"]]'
```

### On the target node

```bash
ssh -i ~/.ssh/id_ed25519 wally@192.168.1.52

# 1. Download GGUF (huggingface-cli is already installed)
mkdir -p ~/models/<model-name>
~/.local/bin/hf download <owner>/<repo> \
  --include '<filename>.gguf' \
  --local-dir ~/models/<model-name>

# 2. Import into Ollama
echo "FROM /home/wally/models/<model-name>/<filename>.gguf" > /tmp/Modelfile
ollama create <ollama-tag> -f /tmp/Modelfile
rm /tmp/Modelfile

# 3. Confirm
ollama list
```

### In the setup script

Add a `hf_pull_and_import` call to the relevant setup script:

```bash
hf_pull_and_import "<ollama-tag>" \
  "<owner>/<repo>" \
  "<filename>.gguf"
```

The function handles: skip if already in Ollama, skip download if GGUF already on disk, import via Modelfile.

---

## Step 2 — Update models.yaml

Add an entry under the appropriate tier (`common`, `mid_tier`, or `large`) in `config/models.yaml`:

```yaml
- name: <display-name>
  ollama_tag: <tag>
  source: ollama          # or: hf
  size_gb: <approx>
  # if source: hf, also add:
  hf_repo: <owner>/<repo>
  hf_file: <filename>.gguf
  # if large model, restrict to capable nodes:
  nodes: [bosgame-m5, lenovo-gb10]
```

Then add the model name to `node_models` for each node it should run on.

---

## Step 3 — Commit and push

```bash
git add config/models.yaml scripts/setup/
git commit -m "add <model-name> to <node>"
git push
```

---

## Sizing reference

| Params | Q4_K_M size | Fits on |
|--------|-------------|---------|
| 7B | ~5 GB | all nodes |
| 14B | ~9 GB | all nodes |
| 27B | ~16 GB | all nodes |
| 32B | ~20 GB | nvidia-ai (40GB VRAM), unified nodes |
| 35B MoE | ~23 GB | all nodes |
| 70B | ~42 GB | unified memory nodes only |
| 80B MoE | ~51 GB | unified memory nodes only |
| 122B MoE | ~81 GB | unified memory nodes only |

## Notes

- **Ollama tag mismatch**: always verify the exact tag with `ollama search` — tags like `qwen3.5:72b` may not exist even if the model does
- **HF rate limits**: unauthenticated downloads are slower; set `HF_TOKEN` env var if you hit limits
- **After import**: the GGUF stays in `~/models/` and Ollama keeps its own copy in `~/ollama-models/` — both can be deleted if space is needed, but you'd need to re-import
- **SSH key**: one-time setup per node — `ssh-copy-id -i ~/.ssh/id_ed25519 <user>@<host>`
