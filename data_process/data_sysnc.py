from huggingface_hub import HfApi

api = HfApi("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")

api.upload_folder(
    folder_path="data/Surge_Frames/AutoLaparo",
    repo_id="KimWu1994/Surgical-Jepa",
    repo_type="dataset",
)


