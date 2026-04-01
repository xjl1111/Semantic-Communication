from pathlib import Path
from huggingface_hub import hf_hub_download

local_dir = Path(__file__).resolve().parents[1] / "data" / "assets" / "downloaded_models" / "ram" / "pretrained"
local_dir.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(
    repo_id="xinyu1205/recognize-anything",
    repo_type="space",
    filename="ram_swin_large_14m.pth",
    local_dir=str(local_dir),
)
print(path)
