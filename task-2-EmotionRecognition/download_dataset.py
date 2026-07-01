import os
import urllib.request
import zipfile
import sys
import time

URL = "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content"
ZIP_PATH = "Audio_Speech_Actors_01-24.zip"
EXTRACT_DIR = "data"

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading: {percent}% | {progress_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB | {speed} KB/s | {duration:.1f}s")
    sys.stdout.flush()

def main():
    print(f"Downloading dataset from: {URL}")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    # Download file
    try:
        urllib.request.urlretrieve(URL, ZIP_PATH, reporthook)
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nError downloading file: {e}")
        sys.exit(1)
        
    # Extract file
    print(f"Extracting zip file to: {EXTRACT_DIR}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction completed successfully!")
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        sys.exit(1)
        
    # Clean up
    print("Cleaning up zip file...")
    try:
        os.remove(ZIP_PATH)
        print("Cleanup completed.")
    except Exception as e:
        print(f"Error deleting zip file: {e}")

if __name__ == "__main__":
    main()
