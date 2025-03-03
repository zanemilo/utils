# 📦 Zane's Developer Utilities
**A collection of powerful, reusable utility scripts designed to streamline development, automate tasks, and enhance workflow efficiency.**

---

## 🚀 Overview
This repository contains modular and production-ready utilities for:  
✅ **Logging & Error Handling** – Robust logging setup and custom error-handling decorators.  
✅ **Repository Automation** – Quickly initialize new Git repositories with useful templates.  
✅ **Data Analysis** – Summarization tools for CSV files, statistics computation, and visualizations.  
✅ **Web Development Scaffolding** – Automate project setup with HTML/CSS/JS frameworks and backend support.  
✅ **Machine Learning Experimentation** – Dataset loaders, model training, and evaluation scripts.  
✅ **Web Scraping & Automation** – Dynamic price scraping and automated web interactions.  

Whether you're building software, analyzing data, or automating workflows, these utilities make your development process **faster, cleaner, and more efficient**.

---

## 🔧 Modules & Features
### 📝 Logging & Error Handling
- **Centralized Logging** – Structured logs for debugging and monitoring.
- **Custom Error Handling** – Decorators and retry mechanisms for robust applications.

### 🔥 Repository Initialization
- **Quick Git Repo Setup** – Automates `README.md`, `.gitignore`, and license generation.
- **CLI Support** – Easily initialize a repository with a single command.

### 📊 Data Analysis & Summarization
- **CSV Parsing & Cleaning** – Read, detect numerical columns, and remove unwanted data interactively.
- **Statistical Computation** – Generate mean, median, mode, variance, standard deviation, and correlation matrices.
- **Visualization** – Auto-generate **histograms, boxplots, and scatter plots** for numerical data.

### 🌐 Web Development Scaffolding
- **Project Setup** – Automatically creates a structured web development folder.
- **HTML/CSS/JS Framework Integration** – Optionally includes Bootstrap, Tailwind, Flask, or Express.
- **Server Setup** – Supports both Python (Flask) and Node.js (Express) backend initialization.

### 🤖 Machine Learning Utilities
- **Dataset Loaders** – MNIST, CIFAR-10, and more.
- **Model Training & Evaluation** – Training pipelines, loss tracking, and accuracy computation.
- **Checkpointing** – Save and restore trained models.

### 🔍 Web Scraping & Automation
- **Price Scraping** – Fetches and logs product prices dynamically.
- **Automated Browsing** – Uses Helium to interact with web pages programmatically.

### 📄 PDF Utilities

- **Merge PDFs** – Combine multiple PDFs into one.
- **Split PDFs** – Create individual pages from a larger PDF.
- **Watermarking** – Insert text-based watermarks on each page.
- **Text Extraction** – Extract text (by page) from PDFs.
- **Encryption** – Encrypt PDFs with password protection.

---

## ⚡ Installation
1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/zanemilo/utils.git
cd utils
```
2️⃣ **Set Up a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
3️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## 🛠 Usage
### 🏗 Automate Web Project Setup
```bash
python webdev_setup.py --project-name MyApp --include-server flask --license MIT
```
Creates a fully structured web development project with an optional backend.

### 📊 Summarize Data Files
```bash
python data_summary.py sample.csv --output summary.csv --save-plots
```
Parses CSV data, computes stats, and generates insightful visualizations.

### 📡 Automate Price Scraping
```bash
python price_scraper.py --product "Laptop" --url "https://example.com/product"
```
Logs and tracks product price changes over time.

### 🔥 Run a Machine Learning Experiment
```bash
python ml_utils.py --dataset mnist --epochs 10 --batch_size 32 --lr 0.001 --save_path mnist_model.pth
```
Trains a model on MNIST with adjustable parameters.

### 📄 Handle PDFs
```bash
python pdf_utils.py merge output.pdf file1.pdf file2.pdf
python pdf_utils.py split input.pdf output_folder
python pdf_utils.py watermark input.pdf output.pdf "Watermark Text"
python pdf_utils.py extract input.pdf
python pdf_utils.py encrypt input.pdf output.pdf password
```
Perform PDF merges, splits, watermarking, text extraction, and encryption through a convenient CLI.

---

## 📜 License
This project is licensed under the **MIT License** – free to use, modify, and contribute.

---

## 🌟 Contributions
Feel free to fork, submit PRs, or open issues! 🚀  

---

## 🔗 Connect with Me
[GitHub](https://github.com/zanemilo) • [LinkedIn](https://www.linkedin.com/in/zanedeso)

---

### ✅ Next Steps
- ✅ Add `requirements.txt` (I can generate it if needed!)
- ✅ Refine documentation for each script (if necessary)
- ✅ Consider making the repo public once you're happy with it
