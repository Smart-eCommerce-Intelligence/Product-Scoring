import pandas as pd
import mysql.connector
import numpy as np
import re
import argparse
import json # Though not directly used for input like stores.json, good to have for consistency
from flask import Flask, jsonify, request # Added Flask
import threading # Added for background processing
import os # Added for FLASK_PORT environment variable

# --- Default Database Configurations ---
DEFAULT_DB_HOST = 'localhost'
DEFAULT_DB_USER = 'root'
DEFAULT_DB_PASSWORD = ''
DEFAULT_DB_NAME_SHOPIFY = 'scrap_test'
DEFAULT_DB_NAME_WOOCOMMERCE = 'scrap_test'
DEFAULT_DB_NAME_ANALYSIS = 'scrap_analyse'

# --- Default Analysis Configuration ---
DEFAULT_TOP_K_OVERALL = 20
DEFAULT_FLAGSHIP_PER_STORE = 3
DEFAULT_WEIGHT_AVAILABILITY = 0.6
DEFAULT_WEIGHT_PRICE = 0.4
DEFAULT_DB_BATCH_SIZE = 500

# --- Argument Parsing ---
# These arguments will define the *default* behavior if the script is run directly
# OR the default configuration for the Flask app when it starts.
# The Flask app itself won't re-parse these on every request.
parser = argparse.ArgumentParser(description="Analyze product data from Shopify and WooCommerce, score products, and save results. Can also be run as a Flask API.")
# (Your argparse definitions remain the same as you provided)
parser.add_argument("--db_host", type=str, default=DEFAULT_DB_HOST, help=f"Database host (default: {DEFAULT_DB_HOST})")
parser.add_argument("--db_user", type=str, default=DEFAULT_DB_USER, help=f"Database user (default: {DEFAULT_DB_USER})")
parser.add_argument("--db_password", type=str, default=DEFAULT_DB_PASSWORD, help="Database password (default: empty)")
parser.add_argument("--db_name_shopify", type=str, default=DEFAULT_DB_NAME_SHOPIFY, help=f"Shopify DB name (default: {DEFAULT_DB_NAME_SHOPIFY})")
parser.add_argument("--db_name_woocommerce", type=str, default=DEFAULT_DB_NAME_WOOCOMMERCE, help=f"WooCommerce DB name (default: {DEFAULT_DB_NAME_WOOCOMMERCE})")
parser.add_argument("--db_name_analysis", type=str, default=DEFAULT_DB_NAME_ANALYSIS, help=f"Analysis DB name (default: {DEFAULT_DB_NAME_ANALYSIS})")
parser.add_argument("--top_k_overall", type=int, default=DEFAULT_TOP_K_OVERALL, help=f"Top K overall (default: {DEFAULT_TOP_K_OVERALL})")
parser.add_argument("--flagship_per_store", type=int, default=DEFAULT_FLAGSHIP_PER_STORE, help=f"Flagship K per store (default: {DEFAULT_FLAGSHIP_PER_STORE})")
parser.add_argument("--weight_availability", type=float, default=DEFAULT_WEIGHT_AVAILABILITY, help=f"Availability weight (default: {DEFAULT_WEIGHT_AVAILABILITY})")
parser.add_argument("--weight_price", type=float, default=DEFAULT_WEIGHT_PRICE, help=f"Price weight (default: {DEFAULT_WEIGHT_PRICE})")
parser.add_argument("--db_batch_size", type=int, default=DEFAULT_DB_BATCH_SIZE, help=f"DB batch size (default: {DEFAULT_DB_BATCH_SIZE})")

# Global variable to hold the script's startup arguments
script_args = None

# --- Global Configuration Variables (will be populated from script_args) ---
DB_CONFIG_SHOPIFY = {}
DB_CONFIG_WOOCOMMERCE = {}
DB_CONFIG_ANALYSIS = {}
TOP_K_OVERALL = DEFAULT_TOP_K_OVERALL # Initialize with defaults
FLAGSHIP_PER_STORE = DEFAULT_FLAGSHIP_PER_STORE
WEIGHT_AVAILABILITY = DEFAULT_WEIGHT_AVAILABILITY
WEIGHT_PRICE = DEFAULT_WEIGHT_PRICE
DB_BATCH_SIZE = DEFAULT_DB_BATCH_SIZE

def populate_global_configs(cmd_args):
    """Populates global config variables from parsed command-line arguments."""
    global DB_CONFIG_SHOPIFY, DB_CONFIG_WOOCOMMERCE, DB_CONFIG_ANALYSIS
    global TOP_K_OVERALL, FLAGSHIP_PER_STORE, WEIGHT_AVAILABILITY, WEIGHT_PRICE, DB_BATCH_SIZE

    DB_CONFIG_SHOPIFY = {
        'host': cmd_args.db_host, 'user': cmd_args.db_user, 'password': cmd_args.db_password, 'database': cmd_args.db_name_shopify
    }
    DB_CONFIG_WOOCOMMERCE = {
        'host': cmd_args.db_host, 'user': cmd_args.db_user, 'password': cmd_args.db_password, 'database': cmd_args.db_name_woocommerce
    }
    DB_CONFIG_ANALYSIS = {
        'host': cmd_args.db_host, 'user': cmd_args.db_user, 'password': cmd_args.db_password, 'database': cmd_args.db_name_analysis
    }
    print(f"DEBUG (Global Configs): DB_CONFIG_SHOPIFY set to: {DB_CONFIG_SHOPIFY}")
    print(f"DEBUG (Global Configs): DB_CONFIG_WOOCOMMERCE set to: {DB_CONFIG_WOOCOMMERCE}")
    print(f"DEBUG (Global Configs): DB_CONFIG_ANALYSIS set to: {DB_CONFIG_ANALYSIS}")

    TOP_K_OVERALL = cmd_args.top_k_overall
    FLAGSHIP_PER_STORE = cmd_args.flagship_per_store
    WEIGHT_AVAILABILITY = cmd_args.weight_availability
    WEIGHT_PRICE = cmd_args.weight_price
    DB_BATCH_SIZE = cmd_args.db_batch_size
    print(f"DEBUG (Global Configs): Analysis Params: TOP_K={TOP_K_OVERALL}, FLAGSHIP_K={FLAGSHIP_PER_STORE}, W_AVAIL={WEIGHT_AVAILABILITY}, W_PRICE={WEIGHT_PRICE}, BATCH_SIZE={DB_BATCH_SIZE}")


# --- DB Connection Function (reusable) ---
# This function remains UNCHANGED (it uses the global DB_CONFIG_* variables passed to it)
def db_connect(config, attempt_creation=False):
    # ... (your existing db_connect logic) ...
    db_name = config['database']
    temp_config = config.copy()
    conn_server = None 
    cursor_server = None 
    if attempt_creation:
        if 'database' in temp_config:
            del temp_config['database']
        try:
            conn_server = mysql.connector.connect(**temp_config)
            cursor_server = conn_server.cursor()
            cursor_server.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            print(f"Database '{db_name}' checked/created.")
        except mysql.connector.Error as err:
            print(f"Error during database creation check for '{db_name}': {err}")
        finally:
            if cursor_server: cursor_server.close()
            if conn_server and conn_server.is_connected(): conn_server.close()
    try:
        conn = mysql.connector.connect(**config)
        print(f"Successfully connected to MySQL database: {db_name} on {config.get('host','N/A')}")
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL ({db_name} on {config.get('host','N/A')}): {err}")
        return None


# --- Fetch Functions ---
# These functions remain UNCHANGED (they use the 'conn' object passed to them)
def fetch_shopify_data(conn):
    # ... (your existing fetch_shopify_data logic) ...
    df = pd.DataFrame()
    if not conn: return df
    try:
        query = "SELECT product_url, title, vendor, price, availability, description, category AS product_category, store_name AS source_store_name FROM products WHERE price IS NOT NULL AND title IS NOT NULL AND price != 0"
        df = pd.read_sql(query, conn)
        if not df.empty:
            df['source_platform'] = 'Shopify'
            df['product_tags'] = None 
            df['sku'] = None
        print(f"Fetched {len(df)} products from Shopify.")
    except Exception as e: print(f"Error fetching Shopify data: {e}")
    return df


def fetch_woocommerce_data(conn):
    # ... (your existing fetch_woocommerce_data logic) ...
    df = pd.DataFrame()
    if not conn: return df
    try:
        query = "SELECT product_url, title, price, tag AS product_tags, sku, category AS product_category FROM barefoot_products WHERE price IS NOT NULL AND title IS NOT NULL AND price != 0"
        df = pd.read_sql(query, conn)
        if not df.empty:
            df['source_platform'] = 'WooCommerce'
            df['source_store_name'] = 'Barefoot Buttons'
            df['vendor'] = 'Barefoot Buttons'
            df['availability'] = 'Available' 
            df['description'] = None
        print(f"Fetched {len(df)} products from WooCommerce (Barefoot Buttons).")
    except Exception as e: print(f"Error fetching WooCommerce data: {e}")
    return df

# --- Preprocessing and Scoring Functions ---
# These functions remain UNCHANGED (clean_html, preprocess_combined_data, calculate_attractiveness_score)
# calculate_attractiveness_score will use the global WEIGHT_AVAILABILITY and WEIGHT_PRICE
def clean_html(raw_html):
    # ... (your existing clean_html logic) ...
    if pd.isna(raw_html) or not isinstance(raw_html, str): return ''
    return re.sub(re.compile('<.*?>'), '', raw_html).strip()

def preprocess_combined_data(df):
    # ... (your existing preprocess_combined_data logic) ...
    if df.empty: return df
    print("\n--- Preprocessing Combined Data ---")
    if 'price' in df.columns:
        def clean_and_convert_price(price_val):
            if pd.isna(price_val): return np.nan
            if isinstance(price_val, (int, float)): return float(price_val)
            try:
                cleaned_price = re.sub(r'[^\d\.]', '', str(price_val))
                return float(cleaned_price) if cleaned_price else np.nan
            except ValueError: return np.nan
        df.loc[:, 'price'] = df['price'].apply(clean_and_convert_price)
        df.dropna(subset=['price'], inplace=True)
    if 'availability' in df.columns:
        df.loc[:, 'is_available_numeric'] = df['availability'].apply(lambda x: 1 if isinstance(x, str) and x.lower() == 'available' else 0)
    else: df['is_available_numeric'] = 0
    if 'description' in df.columns:
        df.loc[:, 'description_text'] = df['description'].apply(clean_html)
    else: df['description_text'] = ''
    for col in ['title', 'vendor', 'product_category', 'source_store_name', 'product_tags', 'sku']:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna('N/A')
        else: 
            df[col] = 'N/A'
    print(f"Combined data preprocessing complete. DataFrame shape: {df.shape}")
    return df

def calculate_attractiveness_score(df, w_availability, w_price): # w_availability and w_price are globals
    # ... (your existing calculate_attractiveness_score logic) ...
    if df.empty or not all(col in df.columns for col in ['is_available_numeric', 'price']):
        if not df.empty: df['final_score'] = 0.0 
        print("Skipping scoring: empty DataFrame or missing 'is_available_numeric'/'price'.")
        return df
    print("\n--- Calculating Attractiveness Score ---")
    df['availability_score'] = df['is_available_numeric']
    price_score_col = pd.Series(0.0, index=df.index, name='price_score')
    if df['price'].nunique() > 1 and not df['price'].isnull().all():
        price_min = df['price'].min()
        price_max = df['price'].max()
        if price_max == price_min: price_score_col.loc[:] = 0.5
        else: price_score_col.loc[:] = (price_max - df['price']) / (price_max - price_min)
    elif len(df['price']) > 0 and not df['price'].isnull().all(): price_score_col.loc[:] = 0.5
    df['price_score'] = price_score_col.fillna(0)
    df['final_score'] = (df['availability_score'] * w_availability) + (df['price_score'] * w_price)
    print("Attractiveness scores calculated.")
    return df

# --- Functions to Save Analysis Results to DB ---
# These functions remain UNCHANGED (create_analysis_tables, save_scored_products_to_db, etc.)
# save_scored_products_to_db's batch_size default will use the global DB_BATCH_SIZE
def create_analysis_tables(conn_analysis):
    # ... (your existing create_analysis_tables logic) ...
    if not conn_analysis: return
    cursor = None
    try:
        cursor = conn_analysis.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scored_products (
                id INT AUTO_INCREMENT PRIMARY KEY, product_url VARCHAR(1024) NOT NULL, title VARCHAR(512),
                source_store_name VARCHAR(100), price DECIMAL(10, 2), is_available_numeric TINYINT,
                description_text TEXT, product_category VARCHAR(255), product_tags TEXT, sku VARCHAR(100),
                source_platform VARCHAR(50), availability_score FLOAT, price_score FLOAT, final_score FLOAT,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uni_scored_product_url (product_url(255)),
                INDEX idx_final_score (final_score DESC), INDEX idx_store_platform (source_store_name, source_platform)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS top_k_products_overall (
                rank_overall INT, product_url VARCHAR(1024) NOT NULL, title VARCHAR(512),
                source_store_name VARCHAR(100), final_score FLOAT, source_platform VARCHAR(50),
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (product_url(255))
            ) ENGINE= InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flagship_products_by_store (
                id INT AUTO_INCREMENT PRIMARY KEY, source_store_name VARCHAR(100), store_rank INT,
                product_url VARCHAR(1024) NOT NULL, title VARCHAR(512), final_score FLOAT, source_platform VARCHAR(50),
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_store_product (source_store_name, product_url(255)),
                INDEX idx_store_name_flagship (source_store_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS store_rankings (
                id INT AUTO_INCREMENT PRIMARY KEY, source_store_name VARCHAR(100) UNIQUE,
                avg_product_score FLOAT, max_product_score FLOAT, source_platform VARCHAR(50),
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
        conn_analysis.commit()
        print("Analysis tables checked/created successfully in product_analysis_db.")
    except mysql.connector.Error as err:
        print(f"Error creating analysis tables: {err}")
    finally:
        if cursor: cursor.close()

def save_scored_products_to_db(df, conn_analysis, batch_size=DB_BATCH_SIZE):
    # ... (your existing save_scored_products_to_db logic) ...
    if df.empty or not conn_analysis: return
    print(f"\n--- Saving Scored Products to Database (batch size: {batch_size}) ---")
    cols_for_db = [
        'product_url', 'title', 'source_store_name', 'price', 'is_available_numeric',
        'description_text', 'product_category', 'product_tags', 'sku', 'source_platform',
        'availability_score', 'price_score', 'final_score'
    ]
    for col in cols_for_db:
        if col not in df.columns:
            df[col] = None 
    df_to_save = df[cols_for_db].copy()
    numeric_cols_to_nullify_nans = ['price', 'availability_score', 'price_score', 'final_score']
    for col in numeric_cols_to_nullify_nans:
        if col in df_to_save.columns:
            df_to_save.loc[:, col] = df_to_save[col].astype(object).where(pd.notnull(df_to_save[col]), None)
    text_cols_to_na = ['product_url', 'title', 'source_store_name', 'description_text', 
                       'product_category', 'product_tags', 'sku', 'source_platform']
    for col in text_cols_to_na:
        if col in df_to_save.columns:
             df_to_save.loc[:, col] = df_to_save[col].fillna('N/A').astype(str)
        else: 
             df_to_save[col] = 'N/A'
    sql = """
    INSERT INTO scored_products (product_url, title, source_store_name, price, is_available_numeric,
                                 description_text, product_category, product_tags, sku, source_platform,
                                 availability_score, price_score, final_score)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        title=VALUES(title), source_store_name=VALUES(source_store_name), price=VALUES(price),
        is_available_numeric=VALUES(is_available_numeric), description_text=VALUES(description_text),
        product_category=VALUES(product_category), product_tags=VALUES(product_tags), sku=VALUES(sku),
        source_platform=VALUES(source_platform), availability_score=VALUES(availability_score),
        price_score=VALUES(price_score), final_score=VALUES(final_score),
        analysis_timestamp=CURRENT_TIMESTAMP;
    """
    all_data_tuples = [tuple(x) for x in df_to_save[cols_for_db].to_numpy()]
    total_saved_count = 0
    for i in range(0, len(all_data_tuples), batch_size):
        batch_tuples = all_data_tuples[i:i + batch_size]
        cursor = None
        try:
            if not conn_analysis.is_connected():
                print("Reconnecting to analysis DB for batch...")
                conn_analysis.reconnect(attempts=3, delay=5)
                if not conn_analysis.is_connected():
                    print("Failed to reconnect. Skipping this batch.")
                    continue
            cursor = conn_analysis.cursor()
            cursor.executemany(sql, batch_tuples)
            conn_analysis.commit()
            total_saved_count += len(batch_tuples)
            print(f"Saved/Updated batch of {len(batch_tuples)} products. Total so far: {total_saved_count}")
        except mysql.connector.Error as err:
            print(f"DB Error saving batch of scored products: {err}")
            if conn_analysis.is_connected(): conn_analysis.rollback()
        except Exception as e:
            print(f"General error saving batch of scored products: {e}")
        finally:
            if cursor: cursor.close()
    print(f"Finished saving/updating a total of {total_saved_count} products in 'scored_products' table.")

def save_top_k_to_db(top_k_df, conn_analysis):
    # ... (your existing save_top_k_to_db logic) ...
    if top_k_df.empty or not conn_analysis: return
    print(f"\n--- Saving Top {len(top_k_df)} Products to Database ---")
    cursor = None
    try:
        cursor = conn_analysis.cursor()
        cursor.execute("DELETE FROM top_k_products_overall;") 
        sql = "INSERT INTO top_k_products_overall (rank_overall, product_url, title, source_store_name, final_score, source_platform) VALUES (%s, %s, %s, %s, %s, %s)"
        data_tuples = [(i + 1, row.get('product_url', 'N/A'), row.get('title', 'N/A'), row.get('source_store_name', 'N/A'), row.get('final_score'), row.get('source_platform', 'N/A'))
                       for i, row in top_k_df.reset_index(drop=True).iterrows()]
        cursor.executemany(sql, data_tuples)
        conn_analysis.commit()
        print(f"Saved {len(data_tuples)} products in 'top_k_products_overall' table.")
    except mysql.connector.Error as err: print(f"DB Error saving top_k products: {err}")
    finally: 
        if cursor: cursor.close()

def save_flagship_to_db(flagship_df, conn_analysis):
    # ... (your existing save_flagship_to_db logic) ...
    if flagship_df.empty or not conn_analysis: return
    print(f"\n--- Saving Flagship Products to Database ---")
    cursor = None
    try:
        cursor = conn_analysis.cursor()
        cursor.execute("DELETE FROM flagship_products_by_store;")
        sql = "INSERT INTO flagship_products_by_store (source_store_name, store_rank, product_url, title, final_score, source_platform) VALUES (%s, %s, %s, %s, %s, %s)"
        data_tuples = []
        ranked_flagship_list = []
        if 'source_store_name' in flagship_df.columns:
            for _, group in flagship_df.groupby('source_store_name'):
                group_copy = group.copy() 
                group_copy.loc[:, 'store_rank'] = range(1, len(group_copy) + 1)
                ranked_flagship_list.append(group_copy)
        if not ranked_flagship_list:
            print("No flagship products after ranking to save.")
            return 
        ranked_flagship_df = pd.concat(ranked_flagship_list)
        for _, row in ranked_flagship_df.iterrows():
             data_tuples.append((
                row.get('source_store_name', 'N/A'), row.get('store_rank', 0), row.get('product_url', 'N/A'),
                row.get('title', 'N/A'), row.get('final_score'), row.get('source_platform', 'N/A')
            ))
        cursor.executemany(sql, data_tuples)
        conn_analysis.commit()
        print(f"Saved {len(data_tuples)} products in 'flagship_products_by_store' table.")
    except mysql.connector.Error as err: print(f"DB Error saving flagship products: {err}")
    finally: 
        if cursor: cursor.close()

def save_store_rankings_to_db(avg_scores_series, max_scores_series, platform_map_df, conn_analysis):
    # ... (your existing save_store_rankings_to_db logic) ...
    if (avg_scores_series.empty and max_scores_series.empty) or not conn_analysis: return
    print("\n--- Saving Store Rankings to Database ---")
    cursor = None
    rankings_df = pd.DataFrame({'avg_product_score': avg_scores_series, 'max_product_score': max_scores_series}).reset_index() 
    if not platform_map_df.empty:
        rankings_df = pd.merge(rankings_df, platform_map_df, on='source_store_name', how='left')
    else:
        rankings_df['source_platform'] = 'Unknown'
    if 'source_platform' not in rankings_df.columns:
        rankings_df['source_platform'] = 'Unknown'
    else:
        rankings_df.loc[:, 'source_platform'] = rankings_df['source_platform'].fillna('Unknown')
    try:
        cursor = conn_analysis.cursor()
        cursor.execute("DELETE FROM store_rankings;")
        sql = "INSERT INTO store_rankings (source_store_name, avg_product_score, max_product_score, source_platform) VALUES (%s, %s, %s, %s)"
        data_tuples = [(row['source_store_name'], row.get('avg_product_score'), row.get('max_product_score'), row['source_platform'])
                       for _, row in rankings_df.iterrows()]
        cursor.executemany(sql, data_tuples)
        conn_analysis.commit()
        print(f"Saved {len(data_tuples)} entries in 'store_rankings' table.")
    except mysql.connector.Error as err: print(f"DB Error saving store rankings: {err}")
    finally: 
        if cursor: cursor.close()

# --- Display functions ---
# These functions remain UNCHANGED (display_top_k_products, etc.)
# They will use the global TOP_K_OVERALL and FLAGSHIP_PER_STORE
def display_top_k_products(df, k):
    # ... (your existing display_top_k_products logic) ...
    if df.empty or 'final_score' not in df.columns: return pd.DataFrame()
    print(f"\n--- Top {k} Most Attractive Products (Overall - Combined) ---")
    top_k = df.sort_values(by='final_score', ascending=False).head(k)
    print(top_k[['title', 'source_store_name', 'price', 'is_available_numeric', 'final_score', 'product_url', 'source_platform']])
    return top_k

def display_flagship_products_per_store(df, n_flagship):
    # ... (your existing display_flagship_products_per_store logic) ...
    if df.empty or not all(col in df.columns for col in ['source_store_name', 'final_score']): return pd.DataFrame()
    print(f"\n--- Top {n_flagship} Flagship Products per Store (Combined) ---")
    # Ensure n_flagship is at least 1 for nlargest to work without error if groups are smaller
    n_flagship = max(1, n_flagship) 
    try:
        # Get index of top N for each group
        # Using a lambda that handles groups smaller than n_flagship
        idx = df.groupby('source_store_name')['final_score'].apply(
            lambda x: x.nlargest(n_flagship if len(x) >= n_flagship else len(x)).index
        )
        if idx.empty: # Handle empty index if no groups or no scores
            print("No flagship products to display after grouping/nlargest.")
            return pd.DataFrame()
        
        # Flatten potential MultiIndex from apply (if groups are present)
        if isinstance(idx, pd.MultiIndex):
            flagship_indices = idx.get_level_values(1)
        elif isinstance(idx, pd.Series) and not idx.empty: # if idx is a Series of indices
            flagship_indices = pd.Index([item for sublist in idx.tolist() for item in sublist if sublist is not None])
        elif isinstance(idx, pd.Index):
            flagship_indices = idx
        else:
            print("Flagship indices could not be determined. Displaying empty DataFrame.")
            return pd.DataFrame()

        if flagship_indices.empty:
            print("No flagship products to display (empty indices).")
            flagship_df_result = pd.DataFrame()
        else:
            flagship_df_result = df.loc[flagship_indices]
            if flagship_df_result.empty:
                print("No flagship products to display (DataFrame empty after loc).")
            else:
                print(flagship_df_result[['source_store_name', 'title', 'final_score', 'price', 'product_url', 'source_platform']])
        return flagship_df_result
    except Exception as e:
        print(f"Error during flagship product display: {e}")
        return pd.DataFrame()


def display_store_rankings(df):
    # ... (your existing display_store_rankings logic) ...
    if df.empty or not all(col in df.columns for col in ['source_store_name', 'final_score']): return pd.Series(dtype=float), pd.Series(dtype=float)
    print("\n--- Store Rankings (Combined) ---")
    avg_store_scores = df.groupby('source_store_name')['final_score'].mean().sort_values(ascending=False)
    print("\nRanked by Average Product Score:"); print(avg_store_scores)
    max_store_scores = df.groupby('source_store_name')['final_score'].max().sort_values(ascending=False)
    print("\nRanked by Best Product Score:"); print(max_store_scores)
    return avg_store_scores, max_store_scores


# --- Main Analysis Logic (to be called by Flask) ---
def run_product_analysis_logic():
    """The core logic of your analysis script, refactored from the original __main__ block."""
    print("--- Product Analysis Logic Triggered ---")
    # Global config variables are already set by populate_global_configs() at script startup

    # Connections are established using global DB_CONFIG_* dicts
    conn_shopify = db_connect(DB_CONFIG_SHOPIFY)
    conn_woocommerce = db_connect(DB_CONFIG_WOOCOMMERCE)
    conn_analysis = db_connect(DB_CONFIG_ANALYSIS, attempt_creation=True)

    if not conn_analysis: # conn_analysis is critical
        print("Critical: Cannot connect to Analysis DB for analysis logic. Aborting.")
        return {"status": "error", "message": "Analysis DB connection failed."}
    
    # We can proceed even if one of the source DBs is down, df_ will be empty
    create_analysis_tables(conn_analysis)

    df_shopify_raw = fetch_shopify_data(conn_shopify)
    df_woocommerce_raw = fetch_woocommerce_data(conn_woocommerce)

    if conn_shopify and conn_shopify.is_connected(): conn_shopify.close(); print(f"MySQL connection to {DB_CONFIG_SHOPIFY['database']} closed.")
    if conn_woocommerce and conn_woocommerce.is_connected(): conn_woocommerce.close(); print(f"MySQL connection to {DB_CONFIG_WOOCOMMERCE['database']} closed.")

    expected_cols = ['product_url', 'title', 'vendor', 'price', 'availability',
                     'description', 'product_category', 'source_store_name',
                     'source_platform', 'product_tags', 'sku']
    df_s_list = []
    if not df_shopify_raw.empty:
        # ... (your logic to ensure all expected_cols are present) ...
        for col in expected_cols:
            if col not in df_shopify_raw.columns: df_shopify_raw[col] = None
        df_s_list.append(df_shopify_raw[expected_cols])

    if not df_woocommerce_raw.empty:
        # ... (your logic to ensure all expected_cols are present) ...
        for col in expected_cols:
            if col not in df_woocommerce_raw.columns: df_woocommerce_raw[col] = None
        df_s_list.append(df_woocommerce_raw[expected_cols])

    combined_df = pd.DataFrame(columns=expected_cols)
    if df_s_list:
        combined_df = pd.concat(df_s_list, ignore_index=True)
        combined_df.dropna(how='all', inplace=True)
        if 'product_url' in combined_df.columns:
            combined_df.drop_duplicates(subset=['product_url'], keep='first', inplace=True)
        print(f"\n(Analysis Logic) Combined DataFrame created. Shape: {combined_df.shape}")

        if not combined_df.empty:
            combined_df = preprocess_combined_data(combined_df)
            if not combined_df.empty:
                combined_df = calculate_attractiveness_score(combined_df, WEIGHT_AVAILABILITY, WEIGHT_PRICE)
                top_k_df = display_top_k_products(combined_df, TOP_K_OVERALL)
                flagship_df_display = display_flagship_products_per_store(combined_df, FLAGSHIP_PER_STORE)
                avg_scores, max_scores = display_store_rankings(combined_df)

                save_scored_products_to_db(combined_df, conn_analysis, DB_BATCH_SIZE)
                save_top_k_to_db(top_k_df, conn_analysis)
                save_flagship_to_db(flagship_df_display, conn_analysis)
                platform_map_df = pd.DataFrame()
                if not combined_df.empty and 'source_store_name' in combined_df.columns and 'source_platform' in combined_df.columns:
                    platform_map_df = combined_df.drop_duplicates(subset=['source_store_name'])[['source_store_name', 'source_platform']].set_index('source_store_name')['source_platform'].to_frame()
                save_store_rankings_to_db(avg_scores, max_scores, platform_map_df, conn_analysis)
                analysis_result_message = "Product analysis completed and results saved."
            else: 
                analysis_result_message = "Analysis halted: Combined DataFrame empty after preprocessing."
                print(analysis_result_message)
        else: 
            analysis_result_message = "Analysis halted: Combined DataFrame empty after initial combination."
            print(analysis_result_message)
    else: 
        analysis_result_message = "Analysis halted: No data fetched from any database."
        print(analysis_result_message)

    if conn_analysis and conn_analysis.is_connected():
        conn_analysis.close()
        print(f"MySQL connection to {DB_CONFIG_ANALYSIS['database']} closed.")
    print("\n--- Product Analysis Logic Complete ---")
    return {"status": "success", "message": analysis_result_message}


# --- Flask App Setup ---
app = Flask(__name__)
is_analyzing = False # Flag to prevent concurrent runs
analysis_thread = None

@app.route('/run_analysis', methods=['POST'])
def trigger_product_analysis():
    global is_analyzing, analysis_thread, script_args

    if script_args is None: # Should be set at startup
        return jsonify({"status": "error", "message": "Analyzer script arguments not initialized."}), 500

    if is_analyzing and analysis_thread and analysis_thread.is_alive():
        return jsonify({"status": "busy", "message": "Product analysis is already in progress."}), 429 # HTTP 429 Too Many Requests

    print("API: Received request to run product analysis.")
    is_analyzing = True

    # Run the analysis logic in a background thread
    # The global configs were set once at startup using script_args,
    # so run_product_analysis_logic will use them.
    analysis_thread = threading.Thread(target=run_analysis_with_status_update)
    analysis_thread.start()

    return jsonify({"status": "triggered", "message": "Product analysis process started in background."}), 202 # HTTP 202 Accepted

def run_analysis_with_status_update():
    """Wrapper to run the main analysis and update status."""
    global is_analyzing
    try:
        # The core analysis logic now uses the global config variables
        # which were set up from script_args when the Flask app started.
        run_product_analysis_logic()
    except Exception as e:
        print(f"Exception during product analysis execution thread: {e}")
    finally:
        is_analyzing = False
        print("Product analysis thread finished.")


if __name__ == '__main__':
    script_args = parser.parse_args() # Parse args at script startup
    print(f"Initial analyzer script arguments parsed: {script_args}")
    populate_global_configs(script_args) # Set up global configs from these args

    # Get port from environment variable for Docker, default to 5003 for analyzer
    port = int(os.environ.get("FLASK_PORT", 5003))
    print(f"Starting Product Analyzer Flask API on port {port}")
    # Set debug=False for production/Docker, use_reloader=False if an external reloader is used (like Gunicorn)
    # For simplicity in Docker, debug=False is good.
    app.run(host='0.0.0.0', port=port, debug=False)