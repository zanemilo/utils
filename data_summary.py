#!/usr/bin/env python3
"""
data_summary.py

Author: Zane M Deso
Purpose: A dynamic data summarization module that provides utilities for:
         - Reading CSV files (with encoding/error options) and detecting numerical columns.
         - Computing essential statistics (mean, median, mode, variance, std, min, max) for numerical columns.
         - Performing correlation analysis between numerical columns.
         - Visualizing distributions via histograms, boxplots, and scatterplots.
         - Saving summary results to CSV.
         
This module is designed to be used both interactively and as part of larger data processing pipelines.

Usage Example (interactive):
    import data_summary as ds
    import logger

    # Setup logging early in your application.
    logger.setup_logging()

    # Read CSV file.
    df = ds.read_csv("sample1.csv")

    # Detect numerical columns.
    num_cols = ds.detect_numerical_columns(df)
    print("Numerical columns:", num_cols)

    # Optionally remove unwanted columns interactively.
    df_clean = ds.remove_columns_interactively(df, num_cols.copy())

    # Compute statistics for each numerical column.
    stats = ds.compute_statistics(df_clean)
    print(stats)

    # Perform correlation analysis for selected columns.
    corr = ds.correlation_analysis(df_clean, columns=num_cols)
    print("Correlation Matrix:\n", corr)

    # Save summary to CSV.
    ds.save_summary_to_csv(stats, corr, "summary_output.csv")

    # View or save plots.
    ds.plot_histograms(df_clean, columns=num_cols, save_plots=False)
    ds.plot_boxplots(df_clean, columns=num_cols, save_plots=False)
    ds.plot_scatter(df_clean, x=num_cols[0], y=num_cols[-1], save_plots=False)

Usage Example (CLI):
    $ python data_summary.py sample1.csv --output summary_output.csv --save-plots --scatter Column1 Column2

License: MIT
"""

import os
import logging
import statistics as st
from collections import Counter
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import argparse

from error_handling import handle_errors

# ----------------------------
# Data Ingestion & Column Detection
# ----------------------------

@handle_errors(default_return=None)
def read_csv(file_path: str, encoding: str = "utf-8", **kwargs) -> Optional[pd.DataFrame]:
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        encoding (str): File encoding (default: "utf-8").
        **kwargs: Additional keyword arguments passed to pandas.read_csv.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing CSV data, or None if an error occurs.
    """
    df = pd.read_csv(file_path, encoding=encoding, **kwargs)
    logging.info("CSV file read successfully from %s", file_path)
    return df

def detect_numerical_columns(df: pd.DataFrame) -> List[str]:
    """
    Detects numerical columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        List[str]: A list of column names with numerical data.
    """
    num_cols = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]
    logging.info("Detected numerical columns: %s", num_cols)
    return num_cols

# ----------------------------
# Data Cleaning (Interactive)
# ----------------------------

def remove_columns_interactively(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    Interactively allows a user to remove unwanted numerical columns.
    Matching is case-insensitive.

    Args:
        df (pd.DataFrame): The original DataFrame.
        num_cols (List[str]): List of detected numerical column names.

    Returns:
        pd.DataFrame: Cleaned DataFrame with selected numerical columns.
    """
    cleaned = df.copy()
    removed = 0
    running = True
    while running:
        print(f"Available numerical columns: {num_cols}")
        try:
            answer = input("Enter the name of a column to remove (or press Enter to finish): ").strip()
        except EOFError:
            print("\nInput interrupted. Exiting interactive mode.")
            break

        if answer == "":
            running = False
        else:
            # Case-insensitive matching.
            matching_cols = [col for col in num_cols if col.lower() == answer.lower()]
            if matching_cols:
                col_to_remove = matching_cols[0]
                num_cols.remove(col_to_remove)
                cleaned.drop(columns=[col_to_remove], inplace=True)
                removed += 1
                print(f"Removed column '{col_to_remove}'.")
            else:
                print(f"Column '{answer}' not found. Please choose from: {num_cols}")
    logging.info("Removed %d columns interactively. Remaining columns: %s", removed, num_cols)
    return cleaned

# ----------------------------
# Statistics Computation
# ----------------------------

def compute_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Computes basic statistics for numerical columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping column names to their statistics.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for col in detect_numerical_columns(df):
        try:
            data = df[col].dropna().tolist()
            if not data:
                continue
            avg = st.mean(data)
            med = st.median(data)
            counts = Counter(data)
            mode = counts.most_common(1)[0][0]
            var = st.variance(data) if len(data) > 1 else 0.0
            std = st.stdev(data) if len(data) > 1 else 0.0
            this_min = min(data)
            this_max = max(data)
            stats[col] = {
                'avg': avg,
                'med': med,
                'mode': mode,
                'var': var,
                'std': std,
                'min': this_min,
                'max': this_max
            }
            logging.info("Computed stats for column %s", col)
        except Exception as e:
            logging.error("Error computing statistics for column %s: %s", col, e)
    return stats

# ----------------------------
# Correlation Analysis
# ----------------------------

def correlation_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Computes the correlation matrix for the specified numerical columns.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        columns (Optional[List[str]]): Columns to include. Defaults to all numerical columns.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    if columns is None:
        columns = detect_numerical_columns(df)
    corr_matrix = df[columns].corr(method="pearson")
    logging.info("Computed correlation matrix for columns: %s", columns)
    return corr_matrix

# ----------------------------
# Save Summary Results
# ----------------------------

def save_summary_to_csv(statistics_dict: Dict[str, Dict[str, Any]],
                        correlation_matrix: pd.DataFrame,
                        output_file: str) -> None:
    """
    Saves statistics and correlation matrix to CSV files.

    Args:
        statistics_dict (Dict[str, Dict[str, Any]]): Dictionary of computed statistics.
        correlation_matrix (pd.DataFrame): Correlation matrix DataFrame.
        output_file (str): Base output filename.
    """
    stats_df = pd.DataFrame(statistics_dict).transpose()
    stats_output = os.path.splitext(output_file)[0] + "_stats.csv"
    corr_output = os.path.splitext(output_file)[0] + "_correlation.csv"
    stats_df.to_csv(stats_output)
    correlation_matrix.to_csv(corr_output)
    logging.info("Saved statistics to %s and correlation matrix to %s", stats_output, corr_output)

# ----------------------------
# Plotting Functions
# ----------------------------

def plot_histograms(df: pd.DataFrame, columns: Optional[List[str]] = None,
                    save_plots: bool = False, output_dir: str = "plots") -> None:
    """
    Plots histograms for specified numerical columns using subplots.
    Optionally saves the plot to a file.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        columns (Optional[List[str]]): Columns to plot. Defaults to all numerical columns.
        save_plots (bool): If True, saves the plot; otherwise, displays it interactively.
        output_dir (str): Directory to save plots.
    """
    if columns is None:
        columns = detect_numerical_columns(df)
    num_plots = len(columns)
    if num_plots == 0:
        logging.info("No numerical columns to plot histograms.")
        return
    ncols = 2
    nrows = (num_plots + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, nrows * 4))
    axes = axes.flatten() if num_plots > 1 else [axes]
    for ax, col in zip(axes, columns):
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "histograms.png")
        plt.savefig(plot_path)
        logging.info("Saved histograms to %s", plot_path)
    else:
        plt.show()
    plt.close()

def plot_boxplots(df: pd.DataFrame, columns: Optional[List[str]] = None,
                  save_plots: bool = False, output_file: str = "boxplots.png") -> None:
    """
    Plots boxplots for specified numerical columns.
    Optionally saves the plot to a file.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        columns (Optional[List[str]]): Columns to plot. Defaults to all numerical columns.
        save_plots (bool): If True, saves the plot; otherwise, displays it interactively.
        output_file (str): Output filename for the plot.
    """
    if columns is None:
        columns = detect_numerical_columns(df)
    plt.figure(figsize=(10, 6))
    df.boxplot(column=columns)
    plt.title("Boxplots")
    plt.tight_layout()
    if save_plots:
        plt.savefig(output_file)
        logging.info("Saved boxplots to %s", output_file)
    else:
        plt.show()
    plt.close()

def plot_scatter(df: pd.DataFrame, x: str, y: str,
                 save_plots: bool = False, output_file: str = "scatter.png") -> None:
    """
    Plots a scatterplot for two numerical columns.
    Optionally saves the plot to a file.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis.
        save_plots (bool): If True, saves the plot; otherwise, displays it interactively.
        output_file (str): Output filename for the plot.
    """
    plt.figure()
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Scatterplot of {x} vs {y}")
    plt.tight_layout()
    if save_plots:
        plt.savefig(output_file)
        logging.info("Saved scatterplot to %s", output_file)
    else:
        plt.show()
    plt.close()

# ----------------------------
# Command-Line Interface (CLI)
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic Data Summarization Tool. Reads a CSV, computes stats, correlation, and plots data."
    )
    parser.add_argument("csv_file", type=str, help="Path to the CSV file to summarize")
    parser.add_argument("--output", type=str, default="summary_output.csv", help="Base output filename for summary CSVs")
    parser.add_argument("--encoding", type=str, default="utf-8", help="CSV file encoding (default: utf-8)")
    parser.add_argument("--no-interactive", action="store_true", help="Run non-interactively (skip interactive column removal)")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files instead of displaying interactively")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--scatter", nargs=2, metavar=("X", "Y"), help="Columns to use for scatterplot")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' does not exist.")
        return

    df = read_csv(args.csv_file, encoding=args.encoding)
    if df is None:
        print("Error reading CSV file.")
        return

    num_cols = detect_numerical_columns(df)
    print("Detected numerical columns:", num_cols)
    if not args.no_interactive:
        df_clean = remove_columns_interactively(df, num_cols.copy())
    else:
        df_clean = df.copy()

    stats = compute_statistics(df_clean)
    print("Computed Statistics:")
    for col, stat in stats.items():
        print(f"{col}: {stat}")

    corr_matrix = correlation_analysis(df_clean)
    print("Correlation Matrix:")
    print(corr_matrix)

    save_summary_to_csv(stats, corr_matrix, args.output)

    # Plotting section
    if args.save_plots:
        plot_histograms(df_clean, columns=num_cols, save_plots=True, output_dir=args.plot_dir)
        plot_boxplots(df_clean, columns=num_cols, save_plots=True, output_file=os.path.join(args.plot_dir, "boxplots.png"))
        if args.scatter:
            x_col, y_col = args.scatter
            plot_scatter(df_clean, x=x_col, y=y_col, save_plots=True, output_file=os.path.join(args.plot_dir, "scatter.png"))
    else:
        plot_choice = input("Would you like to view plots? (y/n): ").strip().lower()
        if plot_choice == "y":
            plot_histograms(df_clean, columns=num_cols, save_plots=False)
            plot_boxplots(df_clean, columns=num_cols, save_plots=False)
            if len(num_cols) >= 2:
                x_col = input("Enter the name of the first column for scatter plot: ").strip()
                y_col = input("Enter the name of the second column for scatter plot: ").strip()
                plot_scatter(df_clean, x=x_col, y=y_col, save_plots=False)

if __name__ == "__main__":
    main()
