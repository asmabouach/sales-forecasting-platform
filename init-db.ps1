# init-db.ps1 - Fixed Manual Postgres Initialization for Docker (Windows-friendly)

Write-Host "Starting manual DB + tables creation..."

# Start postgres if not running
docker compose up -d postgres

# Wait for it to be ready
Write-Host "Waiting 15 seconds for Postgres to start..."
Start-Sleep -Seconds 15

# Create databases SAFELY (using conditional check - no IF NOT EXISTS needed)
# We use a query to check existence first

# For sales_sa
$check_sa = docker compose exec -T postgres psql -U airflow -d airflow -t -c "SELECT 1 FROM pg_database WHERE datname = 'sales_sa'"
if ($check_sa -notmatch "1") {
    Write-Host "Creating sales_sa database..."
    docker compose exec -T postgres psql -U airflow -d airflow -c "CREATE DATABASE sales_sa;"
} else {
    Write-Host "sales_sa already exists - skipping creation."
}

# For sales_dw
$check_dw = docker compose exec -T postgres psql -U airflow -d airflow -t -c "SELECT 1 FROM pg_database WHERE datname = 'sales_dw'"
if ($check_dw -notmatch "1") {
    Write-Host "Creating sales_dw database..."
    docker compose exec -T postgres psql -U airflow -d airflow -c "CREATE DATABASE sales_dw;"
} else {
    Write-Host "sales_dw already exists - skipping creation."
}

# Grant privileges (safe to run even if DBs exist)
Write-Host "Granting privileges..."
docker compose exec -T postgres psql -U airflow -d airflow -c "GRANT ALL PRIVILEGES ON DATABASE sales_sa TO airflow;"
docker compose exec -T postgres psql -U airflow -d airflow -c "GRANT ALL PRIVILEGES ON DATABASE sales_dw TO airflow;"

Write-Host "Databases ready!"

# Now create tables in sales_sa
Write-Host "Creating staging tables in sales_sa..."
docker compose exec -T postgres psql -U airflow -d sales_sa -c @"
DROP TABLE IF EXISTS data_sales;
DROP TABLE IF EXISTS client;
DROP TABLE IF EXISTS invoice;
DROP TABLE IF EXISTS date;
DROP TABLE IF EXISTS sales;

CREATE TABLE data_sales (
    data_id SERIAL PRIMARY KEY,
    client_id VARCHAR(50) NOT NULL,
    invoice_number VARCHAR(50) NOT NULL UNIQUE,
    date DATE NOT NULL,
    gross_sales FLOAT NOT NULL DEFAULT 0,
    discount FLOAT NOT NULL DEFAULT 0,
    net_sales FLOAT NOT NULL DEFAULT 0,
    cog FLOAT NOT NULL DEFAULT 0,
    profit FLOAT GENERATED ALWAYS AS (net_sales - cog) STORED,
    vat FLOAT NOT NULL DEFAULT 0,
    stamp_duty FLOAT NOT NULL DEFAULT 0,
    total_amount FLOAT NOT NULL,
    client_category VARCHAR(100) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    is_return VARCHAR(10) NOT NULL,
    client_type VARCHAR(100) NOT NULL
);

CREATE TABLE client (
    client_id VARCHAR(50) PRIMARY KEY
);

CREATE TABLE invoice (
    invoice_number VARCHAR(50) PRIMARY KEY,
    payment_method VARCHAR(50) NOT NULL,
    is_return VARCHAR(10) NOT NULL
);

CREATE TABLE date (
    date_id INT PRIMARY KEY,
    date DATE NOT NULL
);

CREATE TABLE sales (
    sales_id SERIAL PRIMARY KEY,
    client_id VARCHAR(50) NOT NULL,
    invoice_number VARCHAR(50) NOT NULL UNIQUE,
    date_id INT NOT NULL,
    gross_sales DECIMAL(15,3) NOT NULL DEFAULT 0,
    discount DECIMAL(15,3) NOT NULL DEFAULT 0,
    net_sales DECIMAL(15,3) NOT NULL DEFAULT 0,
    cog DECIMAL(15,3) NOT NULL DEFAULT 0,
    profit DECIMAL(15,3) GENERATED ALWAYS AS (net_sales - cog) STORED,
    vat DECIMAL(15,3) NOT NULL DEFAULT 0,
    stamp_duty DECIMAL(15,3) NOT NULL DEFAULT 0,
    total_amount DECIMAL(15,3) NOT NULL,
    client_category VARCHAR(100) NOT NULL,
    FOREIGN KEY (invoice_number) REFERENCES invoice(invoice_number),
    FOREIGN KEY (client_id) REFERENCES client(client_id),
    FOREIGN KEY (date_id) REFERENCES date(date_id)
);
"@

# Now create tables in sales_dw
Write-Host "Creating DW tables in sales_dw..."
docker compose exec -T postgres psql -U airflow -d sales_dw -c @"
DROP TABLE IF EXISTS dim_client;
DROP TABLE IF EXISTS dim_invoice;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS fact_sales;

CREATE TABLE dim_client (
    client_id VARCHAR(50) PRIMARY KEY
);

CREATE TABLE dim_invoice (
    invoice_number VARCHAR(50) PRIMARY KEY,
    payment_method VARCHAR(50) NOT NULL,
    is_return VARCHAR(10) NOT NULL
);

CREATE TABLE dim_date (
    date_id INT PRIMARY KEY,
    date DATE NOT NULL
);

CREATE TABLE fact_sales (
    sales_id SERIAL PRIMARY KEY,
    client_id VARCHAR(50) NOT NULL,
    invoice_number VARCHAR(50) NOT NULL UNIQUE,
    date_id INT NOT NULL,
    gross_sales DECIMAL(15,3) NOT NULL DEFAULT 0,
    discount DECIMAL(15,3) NOT NULL DEFAULT 0,
    net_sales DECIMAL(15,3) NOT NULL DEFAULT 0,
    cog DECIMAL(15,3) NOT NULL DEFAULT 0,
    profit DECIMAL(15,3) GENERATED ALWAYS AS (net_sales - cog) STORED, 
    vat DECIMAL(15,3) NOT NULL DEFAULT 0,
    stamp_duty DECIMAL(15,3) NOT NULL DEFAULT 0,
    total_amount DECIMAL(15,3) NOT NULL,
    client_category VARCHAR(100) NOT NULL,
    FOREIGN KEY (invoice_number) REFERENCES dim_invoice(invoice_number),
    FOREIGN KEY (client_id) REFERENCES dim_client(client_id),
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
);
"@

Write-Host "`nVerification (tables should appear below):"
Write-Host "Tables in sales_sa:"
docker compose exec -u airflow postgres psql -d sales_sa -c "\dt"

Write-Host "`nTables in sales_dw:"
docker compose exec -u airflow postgres psql -d sales_dw -c "\dt"

Write-Host "`nDone! Run 'docker compose up -d' to start Airflow services."
Write-Host "If you see errors above, share them - likely just need more sleep time or check logs with 'docker compose logs postgres'."