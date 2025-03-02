# Utils Toolbox

This repository is a collection of utility components to help streamline development tasks, automate processes, and support machine learning experiments. The repository includes modules for logging, error handling, repository initialization, price scraping, ML experiment setup, and more.

## Table of Contents

- [Utils Toolbox](#utils-toolbox)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Modules](#modules)
    - [Logger](#logger)
    - [Error Handling](#error-handling)
    - [Repository Initialization](#repository-initialization)
    - [Price Scraper](#price-scraper)
    - [ML Utilities](#ml-utilities)
    - [Find Virtual Environments](#find-virtual-environments)
  - [Installation](#installation)
  - [Usage Examples](#usage-examples)
  - [Testing](#testing)
  - [Dependencies](#dependencies)
  - [License](#license)

## Overview

This repository is designed as a toolbox for developers, providing modular and reusable components that can be easily integrated into various projects. Whether you need to set up logging, manage errors, initialize Git repositories, scrape product prices, or run machine learning experiments, this repo has utilities to help you get started quickly.

## Modules

### Logger

A configurable logging setup that outputs to both a rotating file handler and the console.  
Usage:
```python
import logger
logger.setup_logging()
```

### Error Handling

Provides custom exceptions, error handling decorators, context managers, and a retry mechanism to improve robustness in your applications.  
Usage:
```python
from error_handling import handle_errors, ErrorHandler, retry, setup_global_exception_hook
```

### Repository Initialization

A CLI tool to automate Git repository creation, including setting up README and .gitignore files.  
Usage:
```bash
python init_repo.py -n myrepo -b main -r https://github.com/username/myrepo.git
```

### Price Scraper

A dynamic utility for scraping product prices using Helium.  
Usage:
```bash
python price_scraper.py --product "Product Name" --url "https://example.com/product" --selector ".priceSelector" --output "price_log.txt"
```

### ML Utilities

A robust ML module for loading datasets, building models, training, evaluation, and checkpoint management for experiments with MNIST and CIFAR-10.  
Usage:
```python
from ml_utils import load_dataset, build_model, train_model, evaluate_model, save_checkpoint, load_checkpoint

# Example for MNIST
train_loader, test_loader = load_dataset('mnist', batch_size=64)
model = build_model('mnist')
# Define loss and optimizer
import torch.nn as nn
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model = train_model(model, train_loader, criterion, optimizer, epochs=5, save_path='mnist_model.pth')
accuracy = evaluate_model(model, test_loader)
print(f"MNIST Test Accuracy: {accuracy:.2f}%")
```

Also accessible via CLI:
```bash
python ml_utils.py --dataset cifar10 --epochs 10 --batch_size 64 --lr 0.001 --save_path cifar10_model.pth
```

### Find Virtual Environments

A utility script to search for Python virtual environments (by locating `pyvenv.cfg`) starting from a specified root directory.  
Usage:
```bash
python find_venvs.py --root ~/projects
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/utils-toolbox.git
   cd utils-toolbox
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

Refer to the individual module sections above for CLI usage and example code snippets.

## Testing

To run the tests (if available), use:
```bash
pytest
```
Ensure that your virtual environment is activated before running tests.

## Dependencies

- [Helium](https://github.com/mherrmann/helium) (>= 3.0.0)
- [PyTorch](https://pytorch.org) (>= 1.8.0)
- [torchvision](https://pytorch.org/vision/stable/index.html) (>= 0.9.0)
- [matplotlib](https://matplotlib.org) (>= 3.0.0)
- [pytest](https://docs.pytest.org) (>= 7.0.0) (optional for testing)

## License

This project is licensed under the MIT License.
