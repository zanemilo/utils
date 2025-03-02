#!/usr/bin/env python3
"""
finance_utils.py

Author: Zane Deso (Updated)
Purpose: Provides a comprehensive toolbox for financial and stock analysis in Python,
         integrating functionalities such as:
         - Stock data fetching and Excel-based exploration
         - Data visualization and portfolio simulation
         - Price scraping for financial products
         - Organizing data directories
         - Machine learning model training (demo) for financial predictions
         - Data summarization and correlation analysis

Usage Examples:
    1) Stock data processing and portfolio simulation:
         import finance_utils as futils
         tickers = ["AAPL", "MSFT", "GOOGL"]
         futils.process_stock_list(tickers, "stock_data.xlsx", time_frame="1mo")
         df = futils.load_excel_data("stock_data.xlsx", sheet_name="AAPL")
         futils.visualize_data(df, "AAPL", "./visualizations", choice="4")
         simulator = futils.PortfolioSimulator(initial_investment=50000, monthly_contribution=1500)
         simulator.simulate(periods=12)
         simulator.summary_statistics()

    2) Price scraping:
         price = futils.scrape_financial_product_price("Test Product", "https://www.example.com/product",
                                                       ".price", "test_product_price.txt")

    3) Organize a data directory:
         success = futils.organize_finance_data_directory("./test_directory", organize_by="extension", recursive=False)

    4) Data summarization:
         futils.summarize_financial_data("financial_data.csv", output_base="financial_summary.csv", interactive=False)

    5) Train a demo ML model:
         model, acc = futils.train_financial_model("mnist", batch_size=64, epochs=5, lr=0.001, save_path="mnist_model.pth")

License: MIT
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import custom modules from the utils kit
from error_handling import handle_errors
from logger import setup_logging
from price_scraper import scrape_price, log_price
from organize_dir_by_file_type import organize_directory_flexible
from data_summary import read_csv, detect_numerical_columns, compute_statistics, correlation_analysis, save_summary_to_csv
# Import ml_utils functions (for demo purposes)
from ml_utils import load_dataset, build_model, train_model, evaluate_model

# Initialize logging for the module using the custom logger.
setup_logging()

#############
#  SECTION: STOCK DATA FETCHING & PROCESSING
#############

@handle_errors(default_return=None)
def get_stock_info(ticker: str, time_span: str = "5d") -> Optional[Dict[str, Any]]:
    """
    Fetches stock information for a given ticker, including current price, 
    historical data, and financials. Uses yfinance.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice', 'N/A')
    market_cap = info.get('marketCap', 'N/A')
    sector = info.get('sector', 'N/A')
    previous_close = info.get('previousClose', 'N/A')
    hist = stock.history(period=time_span)
    hist.index = hist.index.tz_localize(None)  # Remove timezone
    financials = stock.financials
    return {
        "ticker": ticker,
        "current_price": current_price,
        "market_cap": market_cap,
        "sector": sector,
        "previous_close": previous_close,
        "historical_data": hist,
        "financials": financials,
    }

def process_stock_list(ticker_list: List[str], output_file: str, time_frame: str = "5d") -> None:
    """
    Fetches data for a list of tickers and saves each ticker's historical data 
    to separate sheets in an Excel file.
    """
    if not ticker_list:
        print("Ticker list is empty. Aborting process.")
        return

    all_stock_data = []
    for ticker in ticker_list:
        print(f"Fetching data for {ticker}...")
        stock_data = get_stock_info(ticker, time_frame)
        if stock_data:
            all_stock_data.append(stock_data)
            print(f"Data for {ticker} retrieved successfully.")
        else:
            print(f"Skipping {ticker} due to errors.")
    try:
        output_path = Path(output_file)
        with pd.ExcelWriter(output_path) as writer:
            for stock in all_stock_data:
                stock["historical_data"].to_excel(writer, sheet_name=stock["ticker"])
        print(f"Stock data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")

@handle_errors(default_return=None)
def load_excel_data(file_path: str, sheet_name: str) -> Optional[pd.DataFrame]:
    """
    Loads a single sheet from an Excel file into a pandas DataFrame.
    """
    data = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
    print(f"Data successfully loaded from sheet '{sheet_name}' in {file_path}")
    return data

#############
#  SECTION: VISUALIZATION
#############

def save_plot_to_file(fig: plt.Figure, output_file: str) -> None:
    """
    Saves a matplotlib figure to the specified file path.
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_file}: {e}")

def plot_boxplot(historical_data: pd.DataFrame, ticker: str, output_dir: str) -> None:
    """
    Generates a box plot for stock prices.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    historical_data[['Open', 'High', 'Low', 'Close']].plot(kind='box', ax=ax)
    ax.set_title(f"Box Plot of Prices for {ticker}")
    ax.set_ylabel("Price")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = Path(output_dir) / f"{ticker}_boxplot.png"
    save_plot_to_file(fig, str(output_path))
    plt.close(fig)

def plot_correlation_matrix(historical_data: pd.DataFrame, ticker: str, output_dir: str) -> None:
    """
    Generates a correlation matrix heatmap for the stock data using matplotlib.
    """
    correlation_matrix = historical_data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(correlation_matrix, interpolation='nearest', cmap='coolwarm')
    ax.set_title(f"Correlation Matrix for {ticker}")
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)
    fig.colorbar(cax)
    output_path = Path(output_dir) / f"{ticker}_correlation_matrix.png"
    save_plot_to_file(fig, str(output_path))
    plt.close(fig)

def plot_time_series_with_moving_avg(historical_data: pd.DataFrame, ticker: str, output_dir: str) -> None:
    """
    Plots the stock price time series along with SMA and EMA moving averages.
    """
    historical_data = historical_data.copy()
    historical_data['SMA_10'] = historical_data['Close'].rolling(window=10).mean()
    historical_data['EMA_10'] = historical_data['Close'].ewm(span=10, adjust=False).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    historical_data[['Close', 'SMA_10', 'EMA_10']].plot(ax=ax)
    ax.set_title(f"Stock Price and Moving Averages for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(['Close', 'SMA (10 days)', 'EMA (10 days)'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = Path(output_dir) / f"{ticker}_time_series.png"
    save_plot_to_file(fig, str(output_path))
    plt.close(fig)

def plot_line_chart(historical_data: pd.DataFrame, ticker: str, output_dir: str) -> None:
    """
    Generates a line chart of stock prices.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    historical_data[['Open', 'High', 'Low', 'Close']].plot(ax=ax)
    ax.set_title(f"Line Chart of Prices for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = Path(output_dir) / f"{ticker}_line_chart.png"
    save_plot_to_file(fig, str(output_path))
    plt.close(fig)

def visualize_data(historical_data: pd.DataFrame, ticker: str, output_dir: str, choice: Optional[str] = None) -> None:
    """
    Generates one of several available plots for the provided stock data.
    Choices:
        "1": Box plot
        "2": Correlation matrix heatmap
        "3": Time series with moving averages
        "4": Line chart
    """
    if not choice:
        print("Choose the type of visualization:")
        print("1. Box Plot of Prices")
        print("2. Correlation Matrix Heatmap")
        print("3. Time Series with Moving Averages")
        print("4. Line Chart of Prices")
        choice = input("Enter your choice: ")

    if choice == "1":
        plot_boxplot(historical_data, ticker, output_dir)
    elif choice == "2":
        plot_correlation_matrix(historical_data, ticker, output_dir)
    elif choice == "3":
        plot_time_series_with_moving_avg(historical_data, ticker, output_dir)
    elif choice == "4":
        plot_line_chart(historical_data, ticker, output_dir)
    else:
        print("Invalid choice. No visualization generated.")

#############
#  SECTION: PORTFOLIO SIMULATION
#############

class PortfolioSimulator:
    """
    A simple portfolio simulator that models portfolio growth.
    """
    def __init__(self, initial_investment: float, monthly_contribution: float) -> None:
        self.initial_investment = initial_investment
        self.monthly_contribution = monthly_contribution
        self.history = []
        print(f"Initialized PortfolioSimulator with initial investment ${initial_investment} and monthly contribution ${monthly_contribution}.")

    def simulate(self, periods: int = 12) -> None:
        """
        Simulates portfolio growth over the given number of periods (months).
        """
        investment = self.initial_investment
        self.history = [investment]
        for month in range(1, periods + 1):
            monthly_return = np.random.uniform(-0.02, 0.05)
            investment = (investment + self.monthly_contribution) * (1 + monthly_return)
            self.history.append(investment)
            print(f"Month {month}: Investment value ${investment:.2f}")
    
    def summary_statistics(self) -> None:
        """
        Prints summary statistics of the simulation.
        """
        if not self.history:
            print("No simulation data available. Run simulate() first.")
            return
        total_periods = len(self.history) - 1
        growth = self.history[-1] - self.history[0]
        avg_growth = growth / total_periods if total_periods > 0 else 0
        print(f"Simulation Summary: Total Growth: ${growth:.2f}, Average Monthly Growth: ${avg_growth:.2f}")

#############
#  SECTION: INTEGRATED UTILS FROM OTHER MODULES
#############

def scrape_financial_product_price(product_name: str, url: str, price_selector: str, output_file: str) -> Optional[str]:
    """
    Scrapes a product price using the price_scraper module and logs it.
    """
    price = scrape_price(url, price_selector)
    if price:
        log_price(product_name, price, output_file)
        print(f"Price for {product_name} logged successfully: {price}")
    else:
        print(f"Failed to scrape price for {product_name}.")
    return price

def organize_finance_data_directory(directory: str, organize_by: str = "extension", recursive: bool = False) -> bool:
    """
    Organizes files in the specified directory based on criteria using the organize_dir_by_file_type module.
    """
    return organize_directory_flexible(directory, organize_by=organize_by, recursive=recursive)

def summarize_financial_data(csv_file: str, output_base: str = "summary_output.csv",
                             encoding: str = "utf-8", interactive: bool = True) -> None:
    """
    Reads a CSV file (e.g., with financial data), computes statistics and correlation analysis,
    and saves the summary to CSV files.
    """
    df = read_csv(csv_file, encoding=encoding)
    if df is None:
        print("Error reading CSV file.")
        return
    num_cols = detect_numerical_columns(df)
    print("Detected numerical columns:", num_cols)
    if interactive:
        # Optionally, allow interactive column removal here if desired.
        pass
    stats = compute_statistics(df)
    corr = correlation_analysis(df, columns=num_cols)
    save_summary_to_csv(stats, corr, output_base)
    print("Data summary completed. Summary files generated.")

def train_financial_model(dataset: str, batch_size: int = 64, epochs: Optional[int] = None,
                          lr: float = 0.001, save_path: Optional[str] = None) -> Tuple[Any, float]:
    """
    Trains a machine learning model using ml_utils functions.
    (For demo purposes, uses standard datasets such as 'mnist' or 'cifar10'.)
    """
    if epochs is None:
        epochs = 5 if dataset.lower() == "mnist" else 10
    train_loader, test_loader = load_dataset(dataset, batch_size)
    model = build_model(dataset)
    import torch.nn as nn
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = train_model(model, train_loader, criterion, optimizer, epochs=epochs, save_path=save_path)
    accuracy = evaluate_model(model, test_loader)
    print(f"Trained model on {dataset} with accuracy: {accuracy:.2f}%")
    return model, accuracy

def setup_finance_logging(log_file: Optional[str] = None, log_level: Optional[str] = None) -> None:
    """
    Configures logging for finance utils using the custom logger module.
    """
    setup_logging(log_file=log_file, log_level=log_level)

#############
#  MAIN EXECUTION BLOCK (DEMO)
#############

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finance Utils Integrated Demo")
    parser.add_argument("--mode", type=str,
                        choices=["stocks", "price", "organize", "summary", "ml"],
                        default="stocks",
                        help=("Select mode:\n"
                              " 'stocks'  - Process stock data, visualize, and simulate portfolio.\n"
                              " 'price'   - Scrape and log product price.\n"
                              " 'organize'- Organize a data directory by file type.\n"
                              " 'summary' - Summarize financial CSV data.\n"
                              " 'ml'      - Train a demo ML model (e.g., on MNIST)."))
    args = parser.parse_args()

    if args.mode == "stocks":
        # Stock data processing demo
        tickers = ["AAPL", "MSFT", "GOOGL"]
        process_stock_list(tickers, "stock_data.xlsx", time_frame="1mo")
        df = load_excel_data("stock_data.xlsx", tickers[0])
        if df is not None:
            visualize_data(df, tickers[0], "./visualizations", choice="4")
        simulator = PortfolioSimulator(initial_investment=50000, monthly_contribution=1500)
        simulator.simulate(periods=12)
        simulator.summary_statistics()

    elif args.mode == "price":
        # Price scraping demo (using placeholder values)
        product_name = "Test Product"
        url = "https://www.example.com/product"
        price_selector = ".price"
        output_file = "test_product_price.txt"
        scrape_financial_product_price(product_name, url, price_selector, output_file)

    elif args.mode == "organize":
        # Directory organization demo
        directory = "./test_directory"  # Ensure this directory exists with some files.
        success = organize_finance_data_directory(directory, organize_by="extension", recursive=False)
        if success:
            print("Directory organized successfully.")
        else:
            print("Directory organization failed. Check logs for details.")

    elif args.mode == "summary":
        # Data summarization demo
        csv_file = "financial_data.csv"  # Provide a valid CSV file with financial data.
        summarize_financial_data(csv_file, output_base="financial_summary.csv", encoding="utf-8", interactive=False)

    elif args.mode == "ml":
        # Machine learning demo (training on MNIST as an example)
        dataset = "mnist"
        train_financial_model(dataset, batch_size=64, epochs=5, lr=0.001, save_path="mnist_model.pth")
