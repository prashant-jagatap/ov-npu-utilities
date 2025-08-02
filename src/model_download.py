from huggingface_hub import snapshot_download


model_id = "OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov"
local_dir = "./OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov"

snapshot_download(repo_id=model_id, local_dir=local_dir)


