#!/usr/bin/env python3
"""
data_summary.py

Author: Zane M Deso
Purpose: A dynamic data summarization module that provides utilities for:
         - Reading CSV files and detecting numerical columns.
         - Computing essential statistics (mean, median, mode, variance, std, min, max) for numerical columns.
         - Performing correlation analysis between numerical columns.
         - Visualizing distributions via histograms, boxplots, and scatterplots.
         - Saving summary results to CSV.
         
This module is designed to be used both interactively and as part of larger data processing pipelines.

Usage Example:
    import data_summary as ds
    import logger

    # Setup logging early in your application.
    logger.setup_logging()

    # Read CSV file.
    df = ds.read_csv("sample1.csv")

    # Detect numerical columns.
    num_cols = ds.detect_numerical_columns(df)
    print("Numerical columns:", num_cols)

    # Optionally remove unwanted columns (interactive or via parameters).
    df_clean = ds.remove_columns_interactively(df, num_cols)

    # Compute statistics for each numerical column.
    stats = ds.compute_statistics(df_clean)
    print(stats)

    # Perform correlation analysis for selected columns.
    corr = ds.correlation_analysis(df_clean, columns=num_cols)
    print("Correlation Matrix:\n", corr)

    # Save summary to CSV.
    ds.save_summary_to_csv(stats, corr, "summary_output.csv")

    # Plot distributions.
    ds.plot_histograms(df_clean, columns=num_cols)
    ds.plot_boxplots(df_clean, columns=num_cols)
    ds.plot_scatter(df_clean, x=num_cols[0], y=num_cols[-1])

License: MIT
"""

import os
import csv
import logging
import statistics as st
from collections import Counter
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt

from error_handling import handle_errors

# ----------------------------
# Data Ingestion & Column Detection
# ----------------------------

@handle_errors(default_return=None)
def read_csv(file_path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        **kwargs: Additional keyword arguments passed to pandas.read_csv.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data, or None if error.
    """
    df = pd.read_csv(file_path, **kwargs)
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
        print(f"Numerical columns detected: {num_cols}")
        answer = input("Enter the name of a column to remove (or press Enter to finish): ").strip()
        if answer == "":
            running = False
        elif answer in num_cols:
            num_cols.remove(answer)
            cleaned.drop(columns=[answer], inplace=True)
            removed += 1
            print(f"Removed column '{answer}'.")
        else:
            print(f"Column '{answer}' not found. Try again.")
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
        Dict[str, Dict[str, Any]]: A dictionary mapping column names to their stats.
    """
    stats = {}
    for col in detect_numerical_columns(df):
        try:
            data = df[col].dropna().tolist()
            if not data:
                continue
            avg = st.mean(data)
            med = st.median(data)
            # Using Counter to find mode(s); if multimodal, pick first
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
        columns (List[str], optional): Columns to include. Defaults to all numerical columns.

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
    Saves statistics and correlation matrix to a CSV file.

    Args:
        statistics_dict (dict): Dictionary of computed statistics.
        correlation_matrix (pd.DataFrame): Correlation matrix DataFrame.
        output_file (str): Output CSV filename.
    """
    # Convert statistics dictionary to DataFrame.
    stats_df = pd.DataFrame(statistics_dict).transpose()
    # Save both stats and correlation matrix into separate sheets if using Excel,
    # or combine them in CSV (here we'll combine as separate CSV files).
    stats_output = os.path.splitext(output_file)[0] + "_stats.csv"
    corr_output = os.path.splitext(output_file)[0] + "_correlation.csv"
    stats_df.to_csv(stats_output)
    correlation_matrix.to_csv(corr_output)
    logging.info("Saved statistics to %s and correlation matrix to %s", stats_output, corr_output)

# ----------------------------
# Plotting Functions
# ----------------------------

def plot_histograms(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """
    Plots histograms for specified numerical columns.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        columns (List[str], optional): Columns to plot. Defaults to all numerical columns.
    """
    if columns is None:
        columns = detect_numerical_columns(df)
    for col in columns:
        plt.figure()
        df[col].hist(bins=20)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    logging.info("Plotted histograms for columns: %s", columns)

def plot_boxplots(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """
    Plots boxplots for specified numerical columns.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        columns (List[str], optional): Columns to plot. Defaults to all numerical columns.
    """
    if columns is None:
        columns = detect_numerical_columns(df)
    df.boxplot(column=columns)
    plt.title("Boxplots")
    plt.tight_layout()
    plt.show()
    logging.info("Plotted boxplots for columns: %s", columns)

def plot_scatter(df: pd.DataFrame, x: str, y: str) -> None:
    """
    Plots a scatterplot for two numerical columns.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis.
    """
    plt.figure()
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Scatterplot of {x} vs {y}")
    plt.tight_layout()
    plt.show()
    logging.info("Plotted scatterplot for %s vs %s", x, y)

# ----------------------------
# Main Routine for CLI
# ----------------------------

if __name__ == "__main__":
    file_path = input("Enter the CSV file path: ").strip()
    df = read_csv(file_path)
    if df is None:
        print("Error reading CSV file.")
        exit(1)

    print("Detected numerical columns:", detect_numerical_columns(df))
    df_clean = remove_columns_interactively(df, detect_numerical_columns(df))
    stats = compute_statistics(df_clean)
    print("Computed Statistics:")
    for col, stat in stats.items():
        print(f"{col}: {stat}")

    corr_matrix = correlation_analysis(df_clean)
    print("Correlation Matrix:")
    print(corr_matrix)

    output_file = input("Enter the output CSV filename (e.g., summary_output.csv): ").strip()
    save_summary_to_csv(stats, corr_matrix, output_file)

    # Optional plotting.
    plot_choice = input("Would you like to view plots? (y/n): ").strip().lower()
    if plot_choice == "y":
        plot_histograms(df_clean)
        plot_boxplots(df_clean)
        if len(detect_numerical_columns(df_clean)) >= 2:
            x_col = input("Enter the name of the first column for scatter plot: ").strip()
            y_col = input("Enter the name of the second column for scatter plot: ").strip()
            plot_scatter(df_clean, x=x_col, y=y_col)
