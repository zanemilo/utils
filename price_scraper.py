#!/usr/bin/env python3
"""
price_scraper.py

Author: Zane Milo
Created: 2025-03-02
Purpose: Dynamically scrape product prices from web pages using Helium.
         The script navigates to a product page, extracts the price using a provided
         CSS selector, and logs the price along with the current date to a specified file.

Usage:
    python price_scraper.py --product "Pro Tools Perpetual License" \
                            --url "https://www.bestbuy.com/site/avid-pro-tools-perpetual-license-mac-os-windows/6317900.p?skuId=6317900" \
                            --selector ".priceView-hero-price.priceView-customer-price span" \
                            --output "Pro-Tools-Price-Index.txt"

    python price_scraper.py --product "Samsung-Q-Series-9.1.4ch-Soundbar" \
                            --url "https://www.bestbuy.com/site/samsung-q-series-9-1-4ch-wireless-true-dolby-atmos-soundbar-with-q-symphony-and-rear-speakers-titan-black/6535890.p?skuId=6535890" \
                            --selector ".priceView-hero-price.priceView-customer-price span" \
                            --output "Samsung-Q-Series-9.1.4ch-Soundbar-Price-Index.txt"

Requirements:
    - Helium (for browser automation)
    - Firefox installed
    - Internet connectivity

Enhancements:
    - Dynamic handling of product details via CLI arguments.
    - Robust error handling and logging.
    - Reusable design for multiple products.
"""

import argparse
import datetime
import logging
import os
import sys
from helium import start_firefox, S, kill_browser

# Configure basic logging for the utility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def scrape_price(url, price_selector):
    """
    Scrape the price from the provided URL using the specified CSS selector.

    Parameters:
        url (str): The URL of the product page.
        price_selector (str): The CSS selector to locate the price element.

    Returns:
        str: The extracted price as a string if found; otherwise, None.
    """
    try:
        logging.info("Starting browser for URL: %s", url)
        start_firefox(url)
        price = None

        # Check if the price element exists using the provided selector
        if S(price_selector).exists():
            logging.info("Price element found using selector: %s", price_selector)
            price_element = S(price_selector)
            if price_element.exists():
                price = price_element.web_element.text
        else:
            logging.warning("Price element not found using selector: %s", price_selector)
    except Exception as e:
        logging.exception("An error occurred while scraping price: %s", e)
        price = None
    finally:
        # Ensure the browser is closed regardless of success or failure
        kill_browser()
    return price

def log_price(product_name, price, output_file):
    """
    Append the scraped price along with today's date and the product name to the output file.

    Parameters:
        product_name (str): The name of the product.
        price (str): The extracted price.
        output_file (str): Path to the file where the price data should be logged.
    """
    today = datetime.date.today()
    log_entry = f"{product_name}-{today}: {price}"
    try:
        # Ensure the directory for the output file exists
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "a") as file:
            file.write(log_entry + "\n")
        logging.info("Price successfully logged: %s", log_entry)
    except Exception as e:
        logging.exception("Failed to log price to file: %s", e)

def main():
    """
    Parse command-line arguments and perform price scraping and logging.

    Required Arguments:
        --product:   Name of the product.
        --url:       URL of the product page.
        --selector:  CSS selector to locate the price element.
        --output:    File to which the price should be logged.
    """
    parser = argparse.ArgumentParser(
        description="Dynamically scrape and log product prices from web pages."
    )
    parser.add_argument("--product", required=True, help="Name of the product")
    parser.add_argument("--url", required=True, help="URL of the product page")
    parser.add_argument("--selector", required=True, help="CSS selector for the price element")
    parser.add_argument("--output", required=True, help="Output file to log the price")
    args = parser.parse_args()

    # Scrape the price using the provided URL and selector
    price = scrape_price(args.url, args.selector)
    if price:
        log_price(args.product, price, args.output)
        print(f"{args.product} price logged successfully: {price}")
    else:
        print(f"Failed to scrape price for {args.product}. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
