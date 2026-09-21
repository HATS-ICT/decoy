import os
from pathlib import Path

import gdown

ASSETS_DIR = Path(__file__).resolve().parent / "env" / "assets"

def download_decompiled_map():
    """Download the de_dust_2.fbx file from Google Drive to env/assets/"""
    file_id = "1P75FvSoT1C_MT1ebG9U8R0dNQcSOQYUT"
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(ASSETS_DIR / "de_dust_2.fbx")
    
    print(f"Downloading de_dust_2.fbx to {output_path}...")
    
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Successfully downloaded de_dust_2.fbx to {output_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        print("Please check your internet connection and try again.")

def download_map_if_not_exist():
    """Download the de_dust_2.fbx file only if it doesn't already exist"""
    
    # Check if file already exists
    output_path = ASSETS_DIR / "de_dust_2.fbx"
    
    if output_path.exists():
        return True
    else:
        print("de_dust_2.fbx not found, downloading...")
        download_decompiled_map()
        return output_path.exists()

if __name__ == "__main__":
    download_decompiled_map()
