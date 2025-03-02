#!/usr/bin/env python3
"""
organize_dir_flexible.py

Author: Zane M Deso
Purpose: Organize files in a specified directory into subfolders based on flexible criteria.
         Organization modes include:
           - "extension": by file extension (default)
           - "date": by last modified date (YYYY-MM)
           - "size": by file size category (small, medium, large)
         Optionally, the scan can be recursive.
         
Usage:
    from organize_dir_flexible import organize_directory_flexible

    # Organize by file extension non-recursively (default).
    organize_directory_flexible("/path/to/dir")

    # Organize recursively by date.
    organize_directory_flexible("/path/to/dir", organize_by="date", recursive=True)

    # Organize recursively by file size.
    organize_directory_flexible("/path/to/dir", organize_by="size", recursive=True)

When executed directly, the script prompts for a directory and additional options.

License: MIT
"""

import os
import glob
import shutil
import logging
import time
from datetime import datetime

from error_handling import handle_errors

# Setup logging.
LOG_FILE = "organize_dir_flexible.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def is_dir_empty(path: str) -> bool:
    """
    Checks if a directory is empty.
    """
    return os.path.isdir(path) and not any(os.listdir(path))

def delete_empty_dir(path: str) -> bool:
    """
    Deletes an empty directory.
    """
    if is_dir_empty(path):
        try:
            os.rmdir(path)
            logging.info("Deleted empty directory: %s", path)
            return True
        except OSError as e:
            logging.error("Error deleting directory %s: %s", path, e)
    return False

def get_destination_folder(file_path: str, base: str, organize_by: str) -> str:
    """
    Determines the destination subfolder for a file based on the organize_by mode.
    
    Modes:
      - "extension": Uses the file extension.
      - "date": Uses last modified date in "YYYY-MM" format.
      - "size": Categorizes files by size.
    
    Returns:
        The destination folder name (not absolute).
    """
    if organize_by == "extension":
        ext = os.path.splitext(file_path)[1].lstrip(".").lower()
        return ext if ext else "no_extension"
    
    elif organize_by == "date":
        try:
            mod_time = os.path.getmtime(file_path)
            folder = datetime.fromtimestamp(mod_time).strftime("%Y-%m")
            return folder
        except Exception as e:
            logging.error("Error getting date for %s: %s", file_path, e)
            return "unknown_date"
    
    elif organize_by == "size":
        try:
            size_bytes = os.path.getsize(file_path)
            # Define arbitrary thresholds (adjust as needed)
            if size_bytes < 1 * 1024 * 1024:  # less than 1MB
                return "small"
            elif size_bytes < 100 * 1024 * 1024:  # less than 100MB
                return "medium"
            else:
                return "large"
        except Exception as e:
            logging.error("Error getting size for %s: %s", file_path, e)
            return "unknown_size"
    else:
        # Fallback: use extension
        return get_destination_folder(file_path, base, "extension")

def scan_files(path: str, recursive: bool = False) -> list:
    """
    Scans the given directory for files. If recursive is True, scans subdirectories.
    Returns a list of absolute file paths.
    """
    file_list = []
    if recursive:
        for root, _, files in os.walk(path):
            for file in files:
                file_list.append(os.path.join(root, file))
    else:
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isfile(full_path):
                file_list.append(full_path)
    return file_list

@handle_errors(default_return=False)
def organize_directory_flexible(path: str, organize_by: str = "extension", recursive: bool = False) -> bool:
    """
    Organizes files in the given directory (and optionally subdirectories) into subfolders
    based on the organize_by mode.

    Args:
        path (str): Absolute path to the directory.
        organize_by (str): Mode of organization: "extension" (default), "date", or "size".
        recursive (bool): Whether to scan subdirectories.

    Returns:
        bool: True if organization completes successfully, False otherwise.
    """
    if not os.path.isdir(path):
        logging.error("Provided path is not a directory: %s", path)
        return False

    logging.info("Starting organization in %s by %s (recursive=%s)", path, organize_by, recursive)
    
    files = scan_files(path, recursive)
    for file_path in files:
        try:
            dest_folder_name = get_destination_folder(file_path, path, organize_by)
            dest_folder = os.path.join(path, dest_folder_name)
            os.makedirs(dest_folder, exist_ok=True)
            destination = os.path.join(dest_folder, os.path.basename(file_path))
            # Move file if it's not already in its destination folder.
            if os.path.abspath(os.path.dirname(file_path)) != os.path.abspath(dest_folder):
                shutil.move(file_path, destination)
                logging.info("Moved file: %s -> %s", file_path, destination)
        except Exception as e:
            logging.error("Error moving file %s: %s", file_path, e)
    
    # After moving, attempt to delete any empty directories (non-recursively in base path).
    for folder in os.listdir(path):
        full_folder = os.path.join(path, folder)
        if os.path.isdir(full_folder) and is_dir_empty(full_folder):
            delete_empty_dir(full_folder)
    
    logging.info("Organization completed for directory: %s", path)
    return True

if __name__ == "__main__":
    dir_path = input("\nEnter the absolute path to the directory to organize:\n")
    organize_mode = input("Enter organization mode (extension, date, size) [default=extension]: ").strip().lower()
    if organize_mode not in ("extension", "date", "size"):
        organize_mode = "extension"
    recursive_input = input("Scan recursively? (y/n) [default=n]: ").strip().lower()
    recursive = recursive_input == "y"
    
    success = organize_directory_flexible(dir_path, organize_by=organize_mode, recursive=recursive)
    if success:
        print("Directory organization completed successfully.")
    else:
        print("There was an error organizing the directory. Check the log for details.")
