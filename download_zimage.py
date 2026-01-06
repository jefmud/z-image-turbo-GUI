from huggingface_hub import snapshot_download


def main():
    # Download weights directly without instantiating the pipeline to avoid GPU allocation.
    repo_id = "Tongyi-MAI/Z-Image-Turbo"
    cache_dir = "models/z-image"

    local_path = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
        allow_patterns="*",
        resume_download=True,
    )

    print(f"Model files downloaded to '{local_path}'.")


if __name__ == "__main__":
    print("DOWNLOAD MODELS: Might be unecessary and timeconsuming for project...")
    response = input("Type _accept_ to download, anything else to quit >  ")

    if 'accept' in response:
        main()
