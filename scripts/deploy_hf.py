"""Publish the web app to a Hugging Face Space (Docker). Run by .github/workflows/deploy-hf.yml.

    HF_TOKEN=hf_... python scripts/deploy_hf.py              # Space <your HF user>/windgap
    HF_TOKEN=hf_... HF_SPACE=team/windgap python scripts/deploy_hf.py
    python scripts/deploy_hf.py --dry-run                     # only build the folder that would be uploaded

The Space is created private the first time; switch it to public in the Space settings when
you are ready to share it. Each run replaces the Space's files with the current repo.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["Dockerfile", ".dockerignore", "requirements-app.txt", "server.py", "src", "templates", "data", "docs"]
FRONT_MATTER = """---
title: WindGap
emoji: 🌬️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Lost energy of a wind farm during a communication outage
---

"""


def build(dest: Path):
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "~$*"))
        else:
            shutil.copy2(src, dest / name)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (dest / "README.md").write_text(FRONT_MATTER + readme, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build(tmp)
        files = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file())
        print(f"{len(files)} files, {sum((tmp / f).stat().st_size for f in files) / 1e6:.1f} MB")
        if args.dry_run:
            print("\n".join(files[:40]) + ("\n…" if len(files) > 40 else ""))
            return

        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        space = os.environ.get("HF_SPACE") or f"{api.whoami()['name']}/windgap"
        api.create_repo(space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
        sha = os.environ.get("GITHUB_SHA", "")[:7]
        api.upload_folder(
            repo_id=space, repo_type="space", folder_path=str(tmp),
            commit_message=f"Deploy from GitHub {sha}".strip(),
            delete_patterns=["*", "**/*"],      # files removed from the repo disappear from the Space too
        )
        print(f"Deployed to https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
