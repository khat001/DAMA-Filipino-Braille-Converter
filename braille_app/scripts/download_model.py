from huggingface_hub import snapshot_download
from pathlib import Path


def main():
    repo_id = "shibarashii/filipino-braille-detection"
    subdir = "braille_detection_yolov11s/weights"
    local_dir = Path("models/").resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Downloading best.pt from {repo_id}/{subdir} ...")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="model",
        allow_patterns=[f"{subdir}/best.pt"]
    )

    print(f"✅ best.pt downloaded to: {path}")


if __name__ == "__main__":
    main()
