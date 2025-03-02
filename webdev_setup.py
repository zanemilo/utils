#!/usr/bin/env python3
"""
webdev_setup.py

Author: Zane M Deso
Purpose: Quickly scaffold a basic web development project with a flexible and adaptive folder structure.
         This utility creates directories for source code, public assets, configuration, and more.
         It generates starter files (e.g., index.html, a basic server file, style.css, script.js) based on user options.
         Optionally, it can scaffold a basic Node package.json for dependency management.

Usage (CLI):
    $ python webdev_setup.py --project-name MyWebApp --include-server flask --license MIT

Arguments:
    --project-name: Name of the project (also creates a folder with that name).
    --include-server: Optionally include a basic server file. Options: "flask", "node".
    --license: License type to include in a LICENSE file (default: MIT).
    --force: Overwrite existing directories if present (clears contents).
    --readme-template: Optional path to a custom README template.
    --index-template: Optional path to a custom index.html template.
    --include-package: (For Node) Create a basic package.json file.
    
License: MIT
"""

import os
import shutil
import argparse
import logging
from typing import Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_directory(path: str, force: bool = False) -> None:
    """
    Create a directory if it does not exist.
    If it exists and force is True, clear its contents.
    """
    try:
        if os.path.exists(path):
            if force:
                logging.info("Directory %s exists; clearing its contents.", path)
                # Remove all contents of the directory
                for filename in os.listdir(path):
                    file_path = os.path.join(path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            else:
                logging.info("Directory %s already exists.", path)
        else:
            os.makedirs(path)
            logging.info("Created directory: %s", path)
    except Exception as e:
        logging.error("Error creating/clearing directory %s: %s", path, e)
        raise

def create_file(path: str, content: str = "") -> None:
    """
    Create a file at the given path with specified content.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info("Created file: %s", path)
    except Exception as e:
        logging.error("Error creating file %s: %s", path, e)
        raise

def get_file_content(template_path: Optional[str], default_content: str) -> str:
    """
    If a template_path is provided and exists, return its content;
    otherwise return the default content.
    """
    if template_path and os.path.isfile(template_path):
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.warning("Error reading template %s: %s. Using default.", template_path, e)
    return default_content

def create_flask_server(base_dir: str) -> None:
    """
    Creates a basic Flask server file.
    """
    server_content = """from flask import Flask, render_template
app = Flask(__name__, static_folder='public', template_folder='public')

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
"""
    server_file = os.path.join(base_dir, "server.py")
    create_file(server_file, content=server_content)

def create_node_server(base_dir: str) -> None:
    """
    Creates a basic Node.js server file using Express.
    """
    server_content = """const express = require('express');
const app = express();
const path = require('path');

app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
"""
    server_file = os.path.join(base_dir, "server.js")
    create_file(server_file, content=server_content)

def create_package_json(base_dir: str, project_name: str) -> None:
    """
    Creates a basic package.json file for a Node.js project.
    """
    package_content = f"""{{
  "name": "{project_name.lower()}",
  "version": "1.0.0",
  "description": "A scaffolded Node.js web project.",
  "main": "server.js",
  "scripts": {{
    "start": "node server.js"
  }},
  "author": "Zane M Deso",
  "license": "MIT",
  "dependencies": {{
    "express": "^4.17.1"
  }}
}}
"""
    package_file = os.path.join(base_dir, "package.json")
    create_file(package_file, content=package_content)

def scaffold_project(project_name: str,
                     include_server: Optional[str] = None,
                     license_type: str = "MIT",
                     force: bool = False,
                     readme_template: Optional[str] = None,
                     index_template: Optional[str] = None,
                     include_package: bool = False) -> None:
    """
    Scaffolds a basic web development project structure.

    Structure:
        project_name/
            src/           -> source code files (e.g., CSS, JS)
            public/        -> public assets (HTML, images)
            config/        -> configuration files
            tests/         -> test files
            LICENSE        -> license file
            README.md      -> readme file
            server.[py/js] -> basic server file if requested
            package.json   -> (optional, for Node projects)
            src/style.css  -> basic CSS file
            src/script.js  -> basic JS file
    """
    base_dir = os.path.abspath(project_name)
    create_directory(base_dir, force=force)

    # Create subdirectories
    subdirs = ["src", "public", "config", "tests"]
    for sub in subdirs:
        create_directory(os.path.join(base_dir, sub), force=force)

    # Create README.md using template if provided
    default_readme = f"# {project_name}\n\nThis project was scaffolded using webdev_setup.py.\n"
    readme_content = get_file_content(readme_template, default_readme)
    create_file(os.path.join(base_dir, "README.md"), content=readme_content)

    # Create LICENSE
    license_content = f"{license_type} License\n\nCopyright (c) 2025 Zane M Deso."
    create_file(os.path.join(base_dir, "LICENSE"), content=license_content)

    # Create index.html in public/ using template if provided
    default_index = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Welcome to {project_name}</title>
    <link rel="stylesheet" href="../src/style.css">
</head>
<body>
    <h1>Welcome to {project_name}</h1>
    <p>This is your starting point for web development.</p>
    <script src="../src/script.js"></script>
</body>
</html>
"""
    index_content = get_file_content(index_template, default_index)
    create_file(os.path.join(base_dir, "public", "index.html"), content=index_content)

    # Create basic style.css in src/ with escaped curly braces
    default_css = """/* Basic CSS for {project_name} */
body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}}
h1 {{
    color: #333;
}}
""".format(project_name=project_name)
    create_file(os.path.join(base_dir, "src", "style.css"), content=default_css)

    # Create basic script.js in src/ with escaped curly braces
    default_js = """// Basic JavaScript for {project_name}
document.addEventListener('DOMContentLoaded', function() {{
    console.log('Welcome to {project_name}!');
}});
""".format(project_name=project_name)
    create_file(os.path.join(base_dir, "src", "script.js"), content=default_js)

    # Optionally create a server file
    if include_server:
        if include_server.lower() == "flask":
            create_flask_server(base_dir)
        elif include_server.lower() == "node":
            create_node_server(base_dir)
            if include_package:
                create_package_json(base_dir, project_name)
        else:
            logging.warning("Server type '%s' not recognized. Skipping server file.", include_server)

    logging.info("Project scaffold complete: %s", base_dir)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold a basic web development project structure."
    )
    parser.add_argument("--project-name", type=str, required=True, help="Name of the project and folder to create")
    parser.add_argument("--include-server", type=str, choices=["flask", "node"], help="Include a basic server file (flask or node)")
    parser.add_argument("--license", type=str, default="MIT", help="License type for the project (default: MIT)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing directories if present (clears contents)")
    parser.add_argument("--readme-template", type=str, help="Path to a custom README template file")
    parser.add_argument("--index-template", type=str, help="Path to a custom index.html template file")
    parser.add_argument("--include-package", action="store_true", help="For Node projects, create a basic package.json file")
    args = parser.parse_args()

    scaffold_project(args.project_name,
                     include_server=args.include_server,
                     license_type=args.license,
                     force=args.force,
                     readme_template=args.readme_template,
                     index_template=args.index_template,
                     include_package=args.include_package)

if __name__ == "__main__":
    main()
