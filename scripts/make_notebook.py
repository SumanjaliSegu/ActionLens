"""Builds notebooks/robotrace_remote_gpu.ipynb (Colab / Kaggle). Run: python scripts/make_notebook.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cells = [
    ("markdown", """# RoboTrace — remote GPU inference (Colab / Kaggle)

Serves a vision-language model behind the HTTP contract in `docs/REMOTE_API.md`, so the public RoboTrace app
can use a free notebook GPU for annotation.

1. Runtime → GPU (Colab T4 or Kaggle T4/P100). Kaggle: also enable Internet in settings.
2. Run all cells. The last cell prints a public URL and a bearer token.
3. In the RoboTrace deployment, set secrets `ROBOTRACE_REMOTE_URL` and `ROBOTRACE_REMOTE_TOKEN`.

The URL lives only as long as this notebook session. RoboTrace health-checks it and falls back gracefully
when it disappears. Default model: `Qwen/Qwen2-VL-2B-Instruct` (Apache-2.0, ~4.5 GB fp16)."""),
    ("code", "!pip -q install 'transformers>=4.45' accelerate fastapi uvicorn python-multipart pycloudflared opencv-python-headless pillow"),
    ("code", """import os, secrets
os.environ.setdefault("ROBOTRACE_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
os.environ["ROBOTRACE_REMOTE_TOKEN"] = secrets.token_urlsafe(24)
# Fetch server.py from your fork of the repo, or upload remote_inference/server.py to the notebook.
REPO_RAW = ""  # e.g. https://raw.githubusercontent.com/<you>/robotrace/main/remote_inference/server.py
if REPO_RAW:
    import urllib.request; urllib.request.urlretrieve(REPO_RAW, "server.py")
assert os.path.exists("server.py"), "Upload remote_inference/server.py to the notebook working directory"
"""),
    ("code", """import server
server.load_model()   # download + load weights once (~2-4 min on first run)
print("model ready on", server.torch.cuda.get_device_name(0) if server.torch.cuda.is_available() else "CPU")"""),
    ("code", """import threading, uvicorn, time
from pycloudflared import try_cloudflare
threading.Thread(target=lambda: uvicorn.run(server.app, host="0.0.0.0", port=8000, log_level="warning"),
                 daemon=True).start()
time.sleep(3)
tunnel = try_cloudflare(port=8000)
print("ROBOTRACE_REMOTE_URL   =", tunnel.tunnel)
print("ROBOTRACE_REMOTE_TOKEN =", os.environ["ROBOTRACE_REMOTE_TOKEN"])"""),
    ("code", """# Optional self-test against a local video file
import requests
r = requests.get(tunnel.tunnel + "/health", timeout=10); print(r.json())
# with open("some_video.mp4", "rb") as f:
#     print(requests.post(tunnel.tunnel + "/annotate", files={"file": f},
#           headers={"Authorization": "Bearer " + os.environ["ROBOTRACE_REMOTE_TOKEN"]}, timeout=600).json())"""),
    ("code", "# Keep the session alive while you demo\nwhile True:\n    time.sleep(60)"),
]

nb = {"cells": [{"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in cells],
      "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "robotrace_remote_gpu.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
