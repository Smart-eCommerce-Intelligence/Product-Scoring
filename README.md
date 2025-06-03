# E-commerce Product Analyzer

This repository contains a Python script (`analyzer.py`) designed to analyze product data scraped from Shopify and WooCommerce platforms. The primary goal is to identify top-performing or most attractive products based on defined criteria and scoring mechanisms. The analyzed results, including scores and rankings, are stored in a separate MySQL database.

## Project Overview

The analyzer takes product data (previously scraped and stored in MySQL databases by the E-commerce Product Scrapers and performs the following key functions:

1.  **Data Ingestion:** Fetches raw product data from separate Shopify and WooCommerce source databases.
2.  **Data Preprocessing:**
    *   Cleans HTML from descriptions.
    *   Standardizes and converts price data to numeric types.
    *   Maps availability status to a numeric score.
    *   Handles missing values.
3.  **Scoring:**
    *   Calculates an "attractiveness score" for each product based on weighted metrics (e.g., availability, price). The weights are configurable.
4.  **Ranking & Selection:**
    *   Identifies the overall Top-K most attractive products.
    *   Determines flagship products (Top-N per store).
    *   Ranks stores based on average and maximum product scores.
5.  **Data Storage:** Saves all processed data, scores, and rankings into a dedicated analysis database.

This analysis provides valuable insights for business intelligence and decision-making.

## Repository Structure

```
.
├── analyzer/
│   ├── analyzer.py             # Main script for product analysis and scoring
│   ├── Dockerfile              # Dockerfile for containerizing the analyzer
│   └── requirements.txt        # Python dependencies for the analyzer
├── kubernetes_manifests/       # (If applicable, for Kubernetes deployment)
│   ├── db-configmap.yaml       # Example ConfigMap for K8s
│   ├── analyzer-job.yaml       # Example Kubernetes Job manifest
│   └── ...                     # Other relevant K8s manifests
├── Jenkinsfile                 # (If applicable, for CI/CD orchestration)
└── README.md                   # This file
```
*(Adjust the `kubernetes_manifests/` and `Jenkinsfile` parts if they are managed elsewhere or not directly part of this repo's core logic for a standalone user).*

## Prerequisites

*   Python 3.8+
*   MySQL Server accessible by the script
*   Access to existing MySQL databases containing scraped Shopify and WooCommerce product data.
*   Docker (if running via a container)

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Smart-eCommerce-Intelligence/Product-Scoring
cd Product-Scoring
```

### 2. Database Setup

*   **Source Databases:** Ensure that the MySQL databases populated by the Shopify scraper (e.g., `shopify_data` with a `products` table) and WooCommerce scraper (e.g., `web_scraping_db` with a `barefoot_products` table) are accessible.
*   **Analysis Database:** The analyzer script will attempt to create a new database (e.g., `product_analysis_db`) and the necessary tables within it (`scored_products`, `top_k_products_overall`, `flagship_products_by_store`, `store_rankings`) if they don't exist. You'll need to provide database credentials with sufficient privileges.

### 3. Python Environment (if running locally without Docker)

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r analyzer/requirements.txt
```

**`analyzer/requirements.txt` should contain:**

```
pandas
numpy
mysql-connector-python
# scikit-learn (optional, if using for more advanced analysis like clustering)
# re, argparse are part of standard library
```

## Running the Analyzer

### Locally (without Docker)

Navigate to the `analyzer` directory:
```bash
cd analyzer
```
Run the script with appropriate command-line arguments:
```bash
python analyzer.py \
    --db_host <your_db_host> \
    --db_user <your_db_user> \
    --db_password "<your_db_password>" \
    --db_name_shopify <source_shopify_db_name> \
    --db_name_woocommerce <source_woocommerce_db_name> \
    --db_name_analysis <target_analysis_db_name> \
    --top_k_overall 20 \
    --flagship_per_store 3 \
    --weight_availability 0.6 \
    --weight_price 0.4
```
*   Replace placeholders with your actual database credentials and desired database names.
*   Adjust scoring weights and Top-K parameters as needed.
*   If `--db_password` is empty, you can omit it or pass `""`.

### Using Docker

A Dockerfile is provided to containerize the analyzer. This is typically built and run as part of a larger CI/CD pipeline (e.g., using Jenkins and Kubernetes).

**To build the image:**

```bash
cd analyzer
docker build -t product-analyzer:latest .
```

**To run the container (example, connecting to a host MySQL):**

Ensure your MySQL database is accessible from Docker containers. For Minikube or Docker Desktop, `host.docker.internal` (on Mac/Windows) or your host's actual IP usually works for `DB_HOST`.

```bash
docker run -it --rm \
    -e DB_HOST="host.docker.internal" \
    -e DB_USER="your_db_user" \
    -e DB_PASSWORD="your_db_password" \
    -e DB_NAME_SHOPIFY="shopify_data" \
    -e DB_NAME_WOOCOMMERCE="web_scraping_db" \
    -e DB_NAME_ANALYSIS="product_analysis_db" \
    -e TOP_K_OVERALL="20" \
    -e FLAGSHIP_PER_STORE="3" \
    -e WEIGHT_AVAILABILITY="0.6" \
    -e WEIGHT_PRICE="0.4" \
    product-analyzer:latest
```
*   The `CMD` in the Dockerfile is set up to use these environment variables to construct the command-line arguments for the Python script.
*   Adjust environment variables (`-e`) as needed. The Dockerfile defines `ANALYZER_..._DEFAULT` values which are used if runtime environment variables are not set.

## Script Arguments

The `analyzer.py` script accepts various command-line arguments to configure database connections, source/target database names, and analysis parameters. Use the `-h` or `--help` flag to see all available options:

```bash
python analyzer/analyzer.py --help
```

Key configurable parameters include:

*   `--db_host`, `--db_user`, `--db_password`: MySQL connection details.
*   `--db_name_shopify`: Name of the database containing Shopify scraped data.
*   `--db_name_woocommerce`: Name of the database containing WooCommerce scraped data.
*   `--db_name_analysis`: Name of the database where analysis results will be stored.
*   `--top_k_overall`: The number of top products to select overall.
*   `--flagship_per_store`: The number of top products to select for each store.
*   `--weight_availability`: Weight for the availability score component.
*   `--weight_price`: Weight for the price score component.
*   `--db_batch_size`: Batch size for inserting data into the analysis database.

## Output

The script will:

1.  Print analysis summaries to the console (Top-K products, flagship products, store rankings).
2.  Populate the following tables in the specified analysis database:
    *   `scored_products`: Contains all processed products with their individual scores.
    *   `top_k_products_overall`: Lists the overall top K products.
    *   `flagship_products_by_store`: Lists the top N flagship products for each store.
    *   `store_rankings`: Provides average and maximum product scores per store.

## Further Development / Future Enhancements

*   Integrate more advanced machine learning models (e.g., `xgboost`, `lightgbm` mentioned in initial project docs) for predictive scoring or clustering.
*   Expand scoring criteria (e.g., sales data if available, product ratings, traffic).
*   Implement more sophisticated text analysis on product descriptions.
*   Add support for more e-commerce platforms.
*   Develop a BI dashboard (e.g., using Streamlit, Power BI) to visualize the analysis results.

