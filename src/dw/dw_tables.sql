-- Drop existing tables if they exist
DROP TABLE IF EXISTS dim_client;
DROP TABLE IF EXISTS dim_invoice;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS fact_sales;

-- Create dim_client table
CREATE TABLE dim_client (
    client_id VARCHAR(50) PRIMARY KEY
);

-- Create dim_invoice table
CREATE TABLE dim_invoice (
    invoice_number VARCHAR(50) PRIMARY KEY,
    payment_method VARCHAR(50) NOT NULL,
    is_return VARCHAR(10) NOT NULL
);

-- Create dim_date table
CREATE TABLE dim_date (
    date_id INT PRIMARY KEY,
    date DATE NOT NULL
);

-- Create fact_sales table
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
