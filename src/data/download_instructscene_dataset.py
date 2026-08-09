import os
from huggingface_hub import hf_hub_url

# Set download path
DOWNLOAD_DIR = "/media/pitbull/HDD2/scene_generation/dataset"

# Create download directory if it doesn't exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Change current working directory to download directory
os.chdir(DOWNLOAD_DIR)

# Get dataset URL from Hugging Face
url = hf_hub_url(
    repo_id="chenguolin/InstructScene_dataset",
    filename="InstructScene.zip",
    repo_type="dataset"
)

print(f"Starting download: {url}")
print(f"Download path: {DOWNLOAD_DIR}")

# Download file and extract
os.system(f"wget {url} && unzip InstructScene.zip")

print("Download and extraction completed!") 