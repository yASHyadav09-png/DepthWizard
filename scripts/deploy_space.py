"""Deploy DepthWizard to a private HF Space (Vercel + ZeroGPU architecture).

Uses the token already stored locally by `hf auth login` (never read/typed by this
script or this session) via `huggingface_hub`'s ambient auth. Does NOT touch anything
public: both the model repo and the Space are created with `private=True`.

Steps:
    1. Create/update a private MODEL repo, upload config.yaml, meta.json, checkpoints/best.pt.
    2. Create/update a private SPACE repo (sdk=gradio), upload space_dist/ (built by
       scripts/build_space.py -- run that first if space_dist/ is missing or stale).
    3. Request ZeroGPU hardware for the Space.
    4. Set the DW_MODEL_REPO Space *variable* (not secret -- it's just the repo id string,
       not a credential) so app.py knows which model repo to download from.

NOT done here (needs the user's own action on huggingface.co, see the printed instructions):
    the Space's own HF_TOKEN secret, which only the account owner can set safely.

Usage:  python scripts/deploy_space.py
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "20260926-003630_phase2b_partial"
SPACE_DIST = ROOT / "space_dist"

MODEL_REPO = "Tanishq0001/depthwizard-model"
SPACE_REPO = "Tanishq0001/depthwizard-app"


def main() -> None:
    api = HfApi()
    me = api.whoami()
    print(f"authenticated as: {me['name']}")
    assert me["name"] == "Tanishq0001", f"unexpected account: {me['name']}"

    # ---- 1. private model repo: config.yaml, meta.json, checkpoints/best.pt (resumable:
    # skip any file already on the Hub with the exact same size, e.g. after a previous
    # run was interrupted mid-upload)
    print(f"\n== model repo: {MODEL_REPO} ==")
    api.create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
    try:
        remote = {s.rfilename: s.size for s in api.repo_info(MODEL_REPO, repo_type="model",
                                                             files_metadata=True).siblings}
    except Exception:
        remote = {}
    for rel in ("config.yaml", "meta.json", "checkpoints/best.pt"):
        src = RUN_DIR / rel
        assert src.is_file(), f"missing {src}"
        size = src.stat().st_size
        if remote.get(rel) == size:
            print(f"  {rel} already uploaded ({size / 1e6:.1f} MB, size matches) -- skipping.")
            continue
        print(f"  uploading {rel} ({size / 1e6:.1f} MB) ...")
        api.upload_file(path_or_fileobj=str(src), path_in_repo=rel, repo_id=MODEL_REPO,
                        repo_type="model", commit_message=f"upload {rel}")
    print("  done.")

    # ---- 2. private Space repo: the built space_dist/ bundle
    # IMPORTANT (found the hard way): the default hardware (cpu-basic) now requires PRO to
    # create even for a private Space ("Static Spaces are free for everyone, but hosting
    # Gradio and Docker Spaces on free cpu-basic requires a PRO subscription" -- confirmed
    # via a live 402 from the API). The free-tier exception is specifically for ZeroGPU
    # hardware requested AT CREATION, not cpu-basic upgraded afterwards -- so space_hardware
    # must be passed to create_repo() itself, not via a separate request_space_hardware() call.
    print(f"\n== space repo: {SPACE_REPO} ==")
    assert SPACE_DIST.is_dir(), f"{SPACE_DIST} missing -- run scripts/build_space.py first"
    api.create_repo(SPACE_REPO, repo_type="space", private=True, space_sdk="gradio",
                    space_hardware=SpaceHardware.ZERO_A10G, exist_ok=True)
    print("  created/exists (ZeroGPU hardware requested at creation).")
    print("  uploading space_dist/ ...")
    api.upload_folder(folder_path=str(SPACE_DIST), repo_id=SPACE_REPO, repo_type="space",
                      commit_message="deploy: mount FastAPI app on Gradio, ZeroGPU wrap")
    print("  done.")

    # ---- 3. non-secret variable: which model repo to download from
    api.add_space_variable(SPACE_REPO, "DW_MODEL_REPO", MODEL_REPO)
    print(f"  DW_MODEL_REPO = {MODEL_REPO}")

    print(f"""
====================================================================
Uploaded. Space (private): https://huggingface.co/spaces/{SPACE_REPO}
                     API base once built: https://tanishq0001-depthwizard-app.hf.space

ONE MANUAL STEP LEFT (you only, on huggingface.co -- not here):
  The Space needs its OWN token to download your PRIVATE model repo at startup.
  1. https://huggingface.co/settings/tokens -> Create new token -> type READ (not write).
  2. https://huggingface.co/spaces/{SPACE_REPO}/settings -> "Variables and secrets" -> New secret
     name:  HF_TOKEN
     value: (paste the Read token there, on that page -- never here in chat)
  3. Save. The Space restarts automatically.

Tell me once that's saved and I'll watch the build and run the full test pass.
====================================================================
""")


if __name__ == "__main__":
    main()
