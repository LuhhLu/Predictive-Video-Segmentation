import gdown
import zipfile
import os

url1 = "https://drive.google.com/uc?id=1YdnV61YPbzwWMeW6tSWxdgDXNkWvGbFf"
url2 = "https://drive.google.com/uc?id=13U_-WC2_0i4IF3FufdeUpISjIyTYUBUd"

# Download the files
gdown.download(url1, output=None, quiet=False)
gdown.download(url2, output=None, quiet=False)

file1 = "Dataset_Student_V2.zip"
file2 = "hidden_set.zip"

target_dir1 = "./"
target_dir2 = "./Dataset_Student"

# Unzip the first file
with zipfile.ZipFile(file1, 'r') as zip_ref:
    for member in zip_ref.infolist():
        try:
            zip_ref.extract(member, target_dir1)
        except zipfile.BadZipFile:
            print(f"Bad CRC-32 for file {member.filename}. Skipping.")
        except Exception as e:
            print(f"Error extracting {member.filename}: {e}. Skipping.")

# Unzip the second file
with zipfile.ZipFile(file2, 'r') as zip_ref:
    for member in zip_ref.infolist():
        try:
            zip_ref.extract(member, target_dir2)
        except zipfile.BadZipFile:
            print(f"Bad CRC-32 for file {member.filename}. Skipping.")
        except Exception as e:
            print(f"Error extracting {member.filename}: {e}. Skipping.")