-- Drop existing tables if they exist
DROP TABLE IF EXISTS data_sales;
DROP TABLE IF EXISTS client;
DROP TABLE IF EXISTS invoice;
DROP TABLE IF EXISTS date;
DROP TABLE IF EXISTS sales;

-- Create staging tables
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

-- Create client table
CREATE TABLE client (
    client_id VARCHAR(50) PRIMARY KEY
);

-- Create invoice table
CREATE TABLE invoice (
    invoice_number VARCHAR(50) PRIMARY KEY,
    payment_method VARCHAR(50) NOT NULL,
    is_return VARCHAR(10) NOT NULL
);

-- Create date table
CREATE TABLE date (
    date_id INT PRIMARY KEY,
    date DATE NOT NULL
);

-- Create sales table
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
