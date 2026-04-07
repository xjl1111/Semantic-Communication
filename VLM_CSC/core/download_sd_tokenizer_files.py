from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
local_dir = Path(__file__).resolve().parents[1] / "data" / "assets" / "downloaded_models" / "sd15" / "tokenizer"
local_dir.mkdir(parents=True, exist_ok=True)

for filename in ["vocab.json", "merges.txt", "special_tokens_map.json"]:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=f"tokenizer/{filename}",
        local_dir=str(local_dir.parent),
        local_dir_use_symlinks=False,
    )
    print(path)
