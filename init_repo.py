#!/usr/bin/env python3
"""
init_repo.py

Author: Zane Milo
Created: 2025-03-02
Purpose: Initializes a new Git repository with essential files and an initial commit.
         This script automates repository initialization by creating a directory, initializing
         a Git repository, creating a README.md and .gitignore file, staging and committing them,
         renaming the default branch, and optionally adding a remote repository URL.

Usage:
    python init_repo.py -n <repository_name> [-b <branch_name>] [-r <remote_url>]

Arguments:
    -n, --name      Name of the repository (required)
    -b, --branch    Default branch name (default: main)
    -r, --remote    Remote repository URL (optional)

Enhancements:
    - Uses logging to track progress and errors.
    - Provides robust error handling and graceful exits.
    - Includes inline documentation and detailed comments for clarity.
"""

import os
import subprocess
import argparse
import sys
import logging

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def run_command(command, cwd=None):
    """
    Run a shell command in the specified directory and exit if it fails.

    Parameters:
        command (str): The shell command to execute.
        cwd (str, optional): The working directory where the command should be executed.

    Raises:
        SystemExit: Exits the script if the command returns a non-zero status.
    """
    logging.info("Executing command: %s (cwd=%s)", command, cwd or os.getcwd())
    result = subprocess.run(command, cwd=cwd, shell=True)
    if result.returncode != 0:
        logging.error("Command failed with return code: %d", result.returncode)
        sys.exit(result.returncode)

def main():
    """
    Main function to parse arguments and initialize a new Git repository.

    Steps:
    1. Parse command-line arguments.
    2. Create a new directory for the repository.
    3. Initialize a Git repository in the new directory.
    4. Create a README.md and .gitignore file.
    5. Stage and commit the initial files.
    6. Rename the default branch.
    7. Optionally add a remote repository URL.
    8. Log a success message.
    """
    parser = argparse.ArgumentParser(description="Initialize a new Git repository with essential files.")
    parser.add_argument("-n", "--name", required=True, help="Name of the repository")
    parser.add_argument("-b", "--branch", default="main", help="Default branch name (default: main)")
    parser.add_argument("-r", "--remote", help="Remote repository URL (optional)")
    args = parser.parse_args()

    # Determine the repository path relative to the current directory
    repo_path = os.path.join(os.getcwd(), args.name)
    try:
        os.makedirs(repo_path, exist_ok=False)
        logging.info("Created repository directory: %s", repo_path)
    except FileExistsError:
        logging.error("Directory '%s' already exists.", repo_path)
        sys.exit(1)
    except Exception as e:
        logging.exception("Failed to create repository directory '%s': %s", repo_path, e)
        sys.exit(1)

    # Initialize the Git repository
    run_command("git init", cwd=repo_path)

    # Create essential files: README.md and .gitignore
    readme_path = os.path.join(repo_path, "README.md")
    try:
        with open(readme_path, "w") as readme:
            readme.write(f"# {args.name}\n")
        logging.info("Created README.md")
    except Exception as e:
        logging.exception("Failed to create README.md: %s", e)
        sys.exit(1)

    gitignore_path = os.path.join(repo_path, ".gitignore")
    try:
        with open(gitignore_path, "w") as gitignore:
            gitignore.write("# Add files and directories to ignore\n")
        logging.info("Created .gitignore")
    except Exception as e:
        logging.exception("Failed to create .gitignore: %s", e)
        sys.exit(1)

    # Stage and commit the initial files
    run_command("git add README.md .gitignore", cwd=repo_path)
    run_command('git commit -m "Initial commit"', cwd=repo_path)

    # Rename the default branch if necessary
    run_command(f"git branch -M {args.branch}", cwd=repo_path)

    # Add a remote origin if a remote URL is provided
    if args.remote:
        run_command(f"git remote add origin {args.remote}", cwd=repo_path)
        logging.info("Added remote repository: %s", args.remote)

    logging.info("Repository '%s' initialized successfully with branch '%s'.", args.name, args.branch)
    print(f"Repository '{args.name}' initialized successfully with branch '{args.branch}'.")

if __name__ == "__main__":
    main()
