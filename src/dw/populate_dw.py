import os
from pathlib import Path
import yaml
from datetime import datetime
import psycopg2
import logging

# === Load environment and DB config ===
from dotenv import load_dotenv
load_dotenv()

# Load paths
with open(Path(__file__).resolve().parents[2] / "configs" / "paths.yml", "r") as f:
    paths = yaml.safe_load(f)

db_config_path = Path(__file__).resolve().parents[2] / paths["db_config"]
logs_dir = Path(__file__).resolve().parents[2] / paths["logs_dir_dw"]

with open(db_config_path, "r") as f:
    db_config = yaml.safe_load(f)

DW_PARAMS = {
    "host": db_config["dw"]["host"],
    "database": db_config["dw"]["database"],
    "user": db_config["dw"]["user"],
    "password": os.getenv("DW_DB_PASSWORD"),
    "port": db_config["dw"]["port"]
}

# === Logging setup ===
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / f"populate_dw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Connect to DW Postgres DB ===
def connect_dw():
    try:
        conn = psycopg2.connect(**DW_PARAMS)
        logging.info("Connected to DW database.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to DW DB: {e}")
        raise

# ------------------------------------------------------------
# Populate dim_client
# ------------------------------------------------------------
def populate_dim_client(conn, staging_conn):
    try:
        cursor = conn.cursor()
        st_cursor = staging_conn.cursor()

        st_cursor.execute("SELECT DISTINCT client_id FROM client;")
        all_clients = {row[0] for row in st_cursor.fetchall()}

        cursor.execute("SELECT client_id FROM dim_client;")
        existing_clients = {row[0] for row in cursor.fetchall()}

        new_clients = all_clients - existing_clients
        if not new_clients:
            logging.info("No new clients to insert into dim_client.")
        else:
            for cid in new_clients:
                cursor.execute("INSERT INTO dim_client (client_id) VALUES (%s);", (cid,))
            conn.commit()
            logging.info(f"Inserted {len(new_clients)} new clients into dim_client.")
        cursor.close()
        st_cursor.close()
    except Exception as e:
        logging.error(f"Error populating dim_client: {e}")
        raise

# ------------------------------------------------------------
# Populate dim_invoice
# ------------------------------------------------------------
def populate_dim_invoice(conn, staging_conn):
    try:
        cursor = conn.cursor()
        st_cursor = staging_conn.cursor()

        st_cursor.execute("SELECT DISTINCT invoice_number, payment_method, is_return FROM invoice;")
        all_invoices = st_cursor.fetchall()

        cursor.execute("SELECT invoice_number FROM dim_invoice;")
        existing_invoices = {row[0] for row in cursor.fetchall()}

        new_invoices = [inv for inv in all_invoices if inv[0] not in existing_invoices]
        if not new_invoices:
            logging.info("No new invoices to insert into dim_invoice.")
        else:
            for invoice_number, payment_method, is_return in new_invoices:
                cursor.execute("""
                    INSERT INTO dim_invoice (invoice_number, payment_method, is_return)
                    VALUES (%s, %s, %s);
                """, (invoice_number, payment_method, is_return))
            conn.commit()
            logging.info(f"Inserted {len(new_invoices)} new invoices into dim_invoice.")
        cursor.close()
        st_cursor.close()
    except Exception as e:
        logging.error(f"Error populating dim_invoice: {e}")
        raise

# ------------------------------------------------------------
# Populate dim_date
# ------------------------------------------------------------
def populate_dim_date(conn, staging_conn):
    try:
        cursor = conn.cursor()
        st_cursor = staging_conn.cursor()

        st_cursor.execute("SELECT DISTINCT date FROM date;")
        all_dates = {row[0] for row in st_cursor.fetchall()}

        cursor.execute("SELECT date FROM dim_date;")
        existing_dates = {row[0] for row in cursor.fetchall()}

        new_dates = all_dates - existing_dates
        if not new_dates:
            logging.info("No new dates to insert into dim_date.")
        else:
            # Find current max date_id
            cursor.execute("SELECT COALESCE(MAX(date_id),0) FROM dim_date;")
            #max_id = cursor.fetchone()[0]
            # Use the staging date_id directly
            st_cursor.execute("SELECT date_id, date FROM date;")
            all_dates = st_cursor.fetchall()

            cursor.execute("SELECT date_id FROM dim_date;")
            existing_dates = {row[0] for row in cursor.fetchall()}

            new_dates = [d for d in all_dates if d[0] not in existing_dates]

            for date_id, date_value in new_dates:
                cursor.execute("INSERT INTO dim_date (date_id, date) VALUES (%s, %s);", (date_id, date_value))
            conn.commit()
            logging.info(f"Inserted {len(new_dates)} new dates into dim_date.")
        cursor.close()
        st_cursor.close()
    except Exception as e:
        logging.error(f"Error populating dim_date: {e}")
        raise

# ------------------------------------------------------------
# Populate fact_sales
# ------------------------------------------------------------
def populate_fact_sales(conn, staging_conn):
    try:
        cursor = conn.cursor()
        st_cursor = staging_conn.cursor()

        st_cursor.execute("""
            SELECT s.invoice_number, s.client_id, s.date_id, s.gross_sales,
                s.discount, s.net_sales, s.cog, s.vat, s.stamp_duty, s.total_amount, s.client_category
            FROM sales s;
        """)
        all_sales = st_cursor.fetchall()

        cursor.execute("SELECT invoice_number FROM fact_sales;")
        existing_sales = {row[0] for row in cursor.fetchall()}

        new_sales = [s for s in all_sales if s[0] not in existing_sales]
        if not new_sales:
            logging.info("No new sales to insert into fact_sales.")
        else:
            for sale in new_sales:
                invoice_number, client_id, date_id, gross_sales, discount, net_sales, cog, vat, stamp_duty, total_amount, client_category = sale

                cursor.execute("""
                    INSERT INTO fact_sales (
                        client_id, invoice_number, date_id,
                        gross_sales, discount, net_sales, cog,
                        vat, stamp_duty, total_amount, client_category
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    client_id, invoice_number, date_id,
                    gross_sales, discount, net_sales, cog,
                    vat, stamp_duty, total_amount, client_category
                ))
            conn.commit()
            logging.info(f"Inserted {len(new_sales)} new sales into fact_sales.")
        cursor.close()
        st_cursor.close()
    except Exception as e:
        logging.error(f"Error populating fact_sales: {e}")
        raise

# === Main ===
if __name__ == "__main__":
    try:
        dw_conn = connect_dw()
        staging_conn = psycopg2.connect(
            host=os.getenv("SA_DB_HOST"),
            database=os.getenv("SA_DB_NAME"),
            user=os.getenv("SA_DB_USER"),
            password=os.getenv("SA_DB_PASSWORD"),
            port=os.getenv("SA_DB_PORT")
        )
        logging.info("Connected to staging database for reading.")

        populate_dim_client(dw_conn, staging_conn)
        populate_dim_invoice(dw_conn, staging_conn)
        populate_dim_date(dw_conn, staging_conn)
        populate_fact_sales(dw_conn, staging_conn)

        dw_conn.close()
        staging_conn.close()
        logging.info("DW population completed successfully.")

    except Exception as e:
        logging.error(f"DW ETL failed: {e}")
        if 'dw_conn' in locals() and dw_conn:
            dw_conn.close()
        if 'staging_conn' in locals() and staging_conn:
            staging_conn.close()
