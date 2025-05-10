# Zane's Developer Utilities
**A collection of powerful, reusable utility scripts designed to streamline development, automate tasks, and enhance workflow efficiency.**

---

## Overview
This repository contains modular and production-ready utilities for:  
- **Logging & Error Handling** – Robust logging setup and custom error-handling decorators.  
- **Repository Automation** – Quickly initialize new Git repositories with useful templates.  
- **Data Analysis** – Summarization tools for CSV files, statistics computation, and visualizations.  
- **Web Development Scaffolding** – Automate project setup with HTML/CSS/JS frameworks and backend support.  
- **Machine Learning Experimentation** – Dataset loaders, model training, and evaluation scripts.  
- **Web Scraping & Automation** – Dynamic price scraping and automated web interactions.  
- **PDF Utilities** – Merge, split, watermark, encrypt, and extract text from PDFs.  

Whether you're building software, analyzing data, or automating workflows, these utilities make your development process faster, cleaner, and more efficient.

---

## Modules & Features

### Logging & Error Handling
- Centralized logging for debugging and monitoring.
- Custom error handling with decorators and retry mechanisms.

### Repository Initialization
- Automated Git repository setup, including `README.md`, `.gitignore`, and license generation.
- Command-line interface for quick initialization.

### Data Analysis & Summarization
- CSV parsing and cleaning, including detection of numerical columns and interactive data removal.
- Statistical computation: mean, median, mode, variance, standard deviation, and correlation matrices.
- Visualization: automatic generation of histograms, boxplots, and scatter plots for numerical data.

### Web Development Scaffolding
- Automated creation of structured web development folders.
- Optional integration with HTML/CSS/JS frameworks such as Bootstrap, Tailwind, Flask, or Express.
- Backend server setup for Python (Flask) and Node.js (Express).

### Machine Learning Utilities
- Dataset loaders for MNIST, CIFAR-10, and others.
- Model training and evaluation pipelines, including loss tracking and accuracy computation.
- Model checkpointing for saving and restoring trained models.

### Web Scraping & Automation
- Price scraping with dynamic fetching and logging.
- Automated browsing using Helium for programmatic web interactions.

### PDF Utilities
- Merge multiple PDFs into one.
- Split PDFs into individual pages.
- Add text-based watermarks to each page.
- Extract text from PDFs by page.
- Encrypt PDFs with password protection.

---

## Installation

1. **Clone the Repository**  
    ```bash
    git clone https://github.com/zanemilo/utils.git
    cd utils
    ```
2. **Set Up a Virtual Environment**  
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3. **Install Dependencies**  
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Automate Web Project Setup
```bash
python webdev_setup.py --project-name MyApp --include-server flask --license MIT
```
Creates a fully structured web development project with an optional backend.

### Summarize Data Files
```bash
python data_summary.py sample.csv --output summary.csv --save-plots
```
Parses CSV data, computes statistics, and generates visualizations.

### Automate Price Scraping
```bash
python price_scraper.py --product "Laptop" --url "https://example.com/product"
```
Logs and tracks product price changes over time.

### Run a Machine Learning Experiment
```bash
python ml_utils.py --dataset mnist --epochs 10 --batch_size 32 --lr 0.001 --save_path mnist_model.pth
```
Trains a model on MNIST with adjustable parameters.

### Handle PDFs
```bash
python pdf_utils.py merge output.pdf file1.pdf file2.pdf
python pdf_utils.py split input.pdf output_folder
python pdf_utils.py watermark input.pdf output.pdf "Watermark Text"
python pdf_utils.py extract input.pdf
python pdf_utils.py encrypt input.pdf output.pdf password
```
Perform PDF merges, splits, watermarking, text extraction, and encryption through a command-line interface.

---

## License
This project is licensed under the MIT License – free to use, modify, and contribute.

---

## Contributions
Contributions are welcome. Feel free to fork, submit pull requests, or open issues.

---

## Connect
[GitHub](https://github.com/zanemilo) • [LinkedIn](https://www.linkedin.com/in/zanedeso)

---

## Next Steps
- Add `requirements.txt` (can be generated if needed)
- Refine documentation for each script as necessary
- Consider making the repository public once finalized
