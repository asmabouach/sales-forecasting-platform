import os
from pathlib import Path
import yaml
from datetime import datetime
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import logging

# === Load environment and DB config ===
load_dotenv()

# Load paths
with open(Path(__file__).resolve().parents[2] / "configs" / "paths.yml", "r") as f:
    paths = yaml.safe_load(f)

data_path = Path(__file__).resolve().parents[2] / paths["processed_parquet"]
db_config_path = Path(__file__).resolve().parents[2] / paths["db_config"]
logs_dir = Path(__file__).resolve().parents[2] / paths["logs_dir_staging"]

with open(db_config_path, "r") as f:
    db_config = yaml.safe_load(f)

DB_PARAMS = {
    "host": db_config["staging"]["host"],
    "database": db_config["staging"]["database"],
    "user": db_config["staging"]["user"],
    "password": os.getenv("SA_DB_PASSWORD"),
    "port": db_config["staging"]["port"]
}

# === Logging setup ===
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / f"populate_staging_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Connect to Postgres ===
def connect_db():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        logging.info("Connected to staging database.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to staging DB: {e}")
        raise

# === Insert data into data_sales ===
def insert_into_data_sales(conn, df):
    cursor = conn.cursor()

    # Fetch existing invoice_numbers from data_sales
    cursor.execute("SELECT invoice_number FROM data_sales;")
    existing_invoices = {row[0] for row in cursor.fetchall()}
    logging.info(f"Found {len(existing_invoices)} existing invoices in data_sales.")

    # Filter only new rows that don't exist yet
    new_rows = df[~df["Invoice_Number"].isin(existing_invoices)]

    if new_rows.empty:
        logging.info("No new invoices to insert. Data is already up to date.")
        cursor.close()
        return

    insert_query = """
        INSERT INTO data_sales (
            client_id, invoice_number, date, gross_sales, discount, net_sales, cog,
            vat, stamp_duty, total_amount, client_category, payment_method,
            is_return, client_type
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s
        );
    """

    # Insert only new rows
    for _, row in new_rows.iterrows():
        cursor.execute(insert_query, (
            row["Client_ID"],
            row["Invoice_Number"],
            row["Date"],
            row["Gross_Sales"],
            row["Discount"],
            row["Net_Sales"],
            row["CoG"],
            row["VAT"],
            row["Stamp_Duty"],
            row["Total_Amount"],
            row["Client_Category"],
            row["Payment_Method"],
            row["Is_Return"],
            row["Client_Type"]
        ))

    conn.commit()
    cursor.close()
    logging.info(f"Inserted {len(new_rows)} new rows into data_sales.")

# ------------------------------------------------------------
# Populate client table incrementally from data_sales
# ------------------------------------------------------------
def populate_client(conn):
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT client_id FROM data_sales;")
        all_clients = {row[0] for row in cursor.fetchall()}

        cursor.execute("SELECT client_id FROM client;")
        existing_clients = {row[0] for row in cursor.fetchall()}

        new_clients = all_clients - existing_clients
        if not new_clients:
            logging.info("No new clients to insert.")
        else:
            for cid in new_clients:
                cursor.execute("INSERT INTO client (client_id) VALUES (%s);", (cid,))
            conn.commit()
            logging.info(f"Inserted {len(new_clients)} new clients.")
        cursor.close()
    except Exception as e:
        logging.error(f"Error populating client table: {e}")
        raise

# ------------------------------------------------------------
# Populate invoice table incrementally from data_sales
# ------------------------------------------------------------
def populate_invoice(conn):
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT invoice_number, payment_method, is_return FROM data_sales;")
        all_invoices = cursor.fetchall()

        cursor.execute("SELECT invoice_number FROM invoice;")
        existing_invoices = {row[0] for row in cursor.fetchall()}

        new_invoices = [inv for inv in all_invoices if inv[0] not in existing_invoices]
        if not new_invoices:
            logging.info("No new invoices to insert.")
        else:
            for invoice_number, payment_method, is_return in new_invoices:
                cursor.execute("""
                    INSERT INTO invoice (invoice_number, payment_method, is_return)
                    VALUES (%s, %s, %s);
                """, (invoice_number, payment_method, is_return))
            conn.commit()
            logging.info(f"Inserted {len(new_invoices)} new invoices.")
        cursor.close()
    except Exception as e:
        logging.error(f"Error populating invoice table: {e}")
        raise

# ------------------------------------------------------------
# Populate date table incrementally from data_sales
# ------------------------------------------------------------
def populate_date(conn):
    try:
        cursor = conn.cursor()

        # Fetch all distinct dates from data_sales
        cursor.execute("SELECT DISTINCT date FROM data_sales;")
        all_dates = {row[0] for row in cursor.fetchall()}

        # Fetch existing dates from date dimension
        cursor.execute("SELECT date FROM date;")
        existing_dates = {row[0] for row in cursor.fetchall()}

        # Find new dates
        new_dates = all_dates - existing_dates

        if not new_dates:
            logging.info("No new dates to insert.")
        else:
            for d in sorted(new_dates):

                # Convert date → YYYYMMDD
                cursor.execute("SELECT TO_CHAR(%s::date, 'YYYYMMDD');", (d,))
                date_id = int(cursor.fetchone()[0])

                # Insert with YYYMMDD ID
                cursor.execute(
                    "INSERT INTO date (date_id, date) VALUES (%s, %s);",
                    (date_id, d)
                )

            conn.commit()
            logging.info(f"Inserted {len(new_dates)} new dates.")

        cursor.close()

    except Exception as e:
        logging.error(f"Error populating date table: {e}")
        raise

# ------------------------------------------------------------
# Populate sales table incrementally using data_sales
# ------------------------------------------------------------
def populate_sales(conn):
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ds.invoice_number, ds.client_id, ds.date, ds.gross_sales,
                   ds.discount, ds.net_sales, ds.cog, ds.vat, ds.stamp_duty,
                   ds.total_amount, ds.client_category
            FROM data_sales ds;
        """)
        all_sales = cursor.fetchall()

        cursor.execute("SELECT invoice_number FROM sales;")
        existing_sales = {row[0] for row in cursor.fetchall()}

        new_sales = [s for s in all_sales if s[0] not in existing_sales]
        if not new_sales:
            logging.info("No new sales to insert.")
        else:
            for sale in new_sales:
                invoice_number, client_id, date, gross_sales, discount, net_sales, cog, vat, stamp_duty, total_amount, client_category = sale

                cursor.execute("SELECT date_id FROM date WHERE date = %s;", (date,))
                date_id_row = cursor.fetchone()
                date_id = date_id_row[0] if date_id_row else None

                cursor.execute("""
                    INSERT INTO sales (
                        client_id, invoice_number, date_id,
                        gross_sales, discount, net_sales, cog,
                        vat, stamp_duty, total_amount, client_category
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    client_id, invoice_number, date_id, gross_sales, discount,
                    net_sales, cog, vat, stamp_duty, total_amount, client_category
                ))
            conn.commit()
            logging.info(f"Inserted {len(new_sales)} new sales.")
        cursor.close()
    except Exception as e:
        logging.error(f"Error populating sales table: {e}")
        raise

# === Main ===
if __name__ == "__main__":
    try:
        conn = connect_db()

        # === Load the processed Excel file ===
        df = pd.read_parquet(data_path)
        logging.info(f"Loaded {len(df)} rows from {data_path}")

        # === Insert and populate ===
        insert_into_data_sales(conn, df)
        populate_client(conn)
        populate_invoice(conn)
        populate_date(conn)
        populate_sales(conn)
        conn.close()
        logging.info("Staging area population completed successfully.")
    except Exception as e:
        logging.error(f"ETL failed: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            logging.info("Connection closed.")

