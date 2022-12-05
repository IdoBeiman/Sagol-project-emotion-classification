from huggingface_hub import snapshot_download
from huggingface_hub import get_full_repo_name
name= get_full_repo_name("gpt-2")
snapshot_download(name)