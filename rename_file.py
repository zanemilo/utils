#!/usr/bin/env python3
import os
import sys

def rename_files_in_dir(directory):
    # Get a list of all files (ignoring subdirectories)
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()  # Sort files alphabetically to get a consistent order

    for idx, filename in enumerate(files):
        # Split the filename into name and extension
        name, ext = os.path.splitext(filename)
        new_name = f"{idx}{ext}"  # Construct new filename with same extension
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)
        print(f"Renaming '{src}' to '{dst}'")
        os.rename(src, dst)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_files.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    rename_files_in_dir(directory)
