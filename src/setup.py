"""
Download and set up the Bank Marketing dataset from the UCI ML Repository.
Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing

Extracts the following files into the data/ directory:
  - bank.csv               (small version, semicolon-separated)
  - bank-full.csv          (full version, semicolon-separated)
  - bank-additional.csv    (additional features, small)
  - bank-additional-full.csv (additional features, full)
  - bank-names.txt         (feature descriptions)
"""

import io
import os
import urllib.request
import zipfile

DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Files to extract from the inner zip (bank.zip inside the outer zip)
TARGET_FILES = {
    "bank.csv",
    "bank-full.csv",
    "bank-additional.csv",
    "bank-additional-full.csv",
    "bank-names.txt",
}


def download_and_setup():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading dataset from {DOWNLOAD_URL} ...")
    with urllib.request.urlopen(DOWNLOAD_URL) as response:
        outer_zip_bytes = response.read()
    print("Download complete.")

    with zipfile.ZipFile(io.BytesIO(outer_zip_bytes)) as outer_zip:
        inner_zip_names = [n for n in outer_zip.namelist() if n.endswith(".zip")]

        # Extract top-level target files (e.g. bank-names.txt)
        for name in outer_zip.namelist():
            filename = os.path.basename(name)
            if filename in TARGET_FILES:
                dest = os.path.join(DATA_DIR, filename)
                with outer_zip.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                print(f"  Extracted: {filename}")

        # Extract target files from any inner zip (e.g. bank.zip)
        for inner_name in inner_zip_names:
            with outer_zip.open(inner_name) as inner_file:
                with zipfile.ZipFile(io.BytesIO(inner_file.read())) as inner_zip:
                    for name in inner_zip.namelist():
                        filename = os.path.basename(name)
                        if filename in TARGET_FILES:
                            dest = os.path.join(DATA_DIR, filename)
                            with inner_zip.open(name) as src, open(dest, "wb") as dst:
                                dst.write(src.read())
                            print(f"  Extracted: {filename}")

    # Confirm all expected files are present
    missing = TARGET_FILES - set(os.listdir(DATA_DIR))
    if missing:
        print(f"\nWarning: the following files were not found in the archive: {missing}")
    else:
        print(f"\nAll data files are ready in: {os.path.abspath(DATA_DIR)}")


if __name__ == "__main__":
    download_and_setup()
