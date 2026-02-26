import pytest
import pandas as pd
from datetime import date
from unittest.mock import Mock, MagicMock, patch

# Import the functions from your script
from src.staging.populate_staging import (
    insert_into_data_sales,
    populate_client,
    populate_invoice,
    populate_date,
    populate_sales,
)


@pytest.fixture
def sample_df():
    data = {
        'Client_ID': ['C001', 'C001', 'C002'],
        'Invoice_Number': ['INV001', 'INV002', 'INV003'],
        'Date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
        'Gross_Sales': [1000.0, 2000.0, 1500.0],
        'Discount': [100.0, 0.0, 150.0],
        'Net_Sales': [900.0, 2000.0, 1350.0],
        'CoG': [500.0, 1000.0, 750.0],
        'VAT': [162.0, 324.0, 243.0],
        'Stamp_Duty': [0.0, 0.0, 0.0],
        'Total_Amount': [1062.0, 2324.0, 1593.0],
        'Client_Category': ['Retail', 'Wholesale', 'Retail'],
        'Payment_Method': ['Cash', 'Card', 'Cash'],
        'Is_Return': ['No', 'No', 'Yes'],
        'Client_Type': ['Individual', 'Business', 'Individual'],
    }
    return pd.DataFrame(data)


# =============================================
# Test insert_into_data_sales (core deduplication)
# =============================================
def test_insert_into_data_sales_no_duplicates(sample_df):
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value

    # Simulate some existing invoices
    mock_cursor.fetchall.return_value = [('INV001',), ('INV002',)]

    with patch('src.staging.populate_staging.logging') as mock_logging:
        insert_into_data_sales(mock_conn, sample_df)

    # Only INV003 should be inserted → 1 execute call with data
    insert_calls = [call for call in mock_cursor.execute.mock_calls if 'INSERT INTO data_sales' in str(call)]
    assert len(insert_calls) == 1

    # Commit called once
    mock_conn.commit.assert_called_once()

    # Log messages
    mock_logging.info.assert_any_call("Found 2 existing invoices in data_sales.")
    mock_logging.info.assert_any_call("Inserted 1 new rows into data_sales.")


def test_insert_into_data_sales_all_new(sample_df):
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.fetchall.return_value = []  # Nothing exists

    insert_into_data_sales(mock_conn, sample_df)

    # 3 inserts expected
    insert_calls = [call for call in mock_cursor.execute.mock_calls if 'INSERT INTO data_sales' in str(call)]
    assert len(insert_calls) == 3


def test_insert_into_data_sales_no_new(sample_df):
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value

    # All invoices already exist
    mock_cursor.fetchall.return_value = [('INV001',), ('INV002',), ('INV003',)]

    with patch('src.staging.populate_staging.logging') as mock_logging:
        insert_into_data_sales(mock_conn, sample_df)

    # No INSERT calls
    insert_calls = [call for call in mock_cursor.execute.mock_calls if 'INSERT INTO data_sales' in str(call)]
    assert len(insert_calls) == 0

    mock_logging.info.assert_any_call("No new invoices to insert. Data is already up to date.")


# =============================================
# Test populate_client
# =============================================
def test_populate_client():
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value

    # data_sales has C001 and C002
    mock_cursor.fetchall.side_effect = [
        [('C001',), ('C001',), ('C002',)],  # DISTINCT client_id from data_sales
        [('C001',)],                        # existing in client table
    ]

    populate_client(mock_conn)

    # One new client (C002) inserted
    insert_calls = [call for call in mock_cursor.execute.mock_calls if 'INSERT INTO client' in str(call)]
    assert len(insert_calls) == 1
    mock_conn.commit.assert_called_once()


# =============================================
# Test populate_invoice
# =============================================
def test_populate_invoice():
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value

    mock_cursor.fetchall.side_effect = [
        [('INV001', 'Cash', 'No'), ('INV002', 'Card', 'No'), ('INV003', 'Cash', 'Yes')],
        [('INV001',)],  # only INV001 exists
    ]

    populate_invoice(mock_conn)

    insert_calls = [call for call in mock_cursor.execute.mock_calls if 'INSERT INTO invoice' in str(call)]
    assert len(insert_calls) == 2  # INV002 and INV003


# =============================================
# Test populate_date
# =============================================
def test_populate_date():
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value

    # Three distinct dates in data_sales, none exist yet
    mock_cursor.fetchall.side_effect = [
        [(date(2025,1,1),), (date(2025,1,2),), (date(2025,1,3),)],
        [],  # no existing dates
    ]

    # Mock the TO_CHAR query that generates date_id
    mock_cursor.fetchone.side_effect = [
        ('20250101',), ('20250102',), ('20250103',)
    ]

    populate_date(mock_conn)

    insert_calls = [call for call in mock_cursor.execute.mock_calls if 'INSERT INTO date' in str(call)]
    assert len(insert_calls) == 3

    # Check one example: date_id is int and correct
    expected_call = ('INSERT INTO date (date_id, date) VALUES (%s, %s);', (20250102, date(2025,1,2)))
    assert any(expected_call == call.args for call in insert_calls)


# =============================================
# Test populate_sales
# =============================================
def test_populate_sales():
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value

    # Simulate 3 sales rows from data_sales, none exist in sales table
    mock_cursor.fetchall.side_effect = [
        [  # SELECT from data_sales
            ('INV001', 'C001', date(2025,1,1), 1000.0, 100.0, 900.0, 500.0, 162.0, 0.0, 1062.0, 'Retail'),
            ('INV002', 'C001', date(2025,1,2), 2000.0, 0.0, 2000.0, 1000.0, 324.0, 0.0, 2324.0, 'Wholesale'),
            ('INV003', 'C002', date(2025,1,3), 1500.0, 150.0, 1350.0, 750.0, 243.0, 0.0, 1593.0, 'Retail'),
        ],
        [],  # no existing invoice_numbers in sales table
    ]

    # Mock date_id lookup → returns integer in a tuple (as fetchone does)
    mock_cursor.fetchone.side_effect = [(20250101,), (20250102,), (20250103,)]

    populate_sales(mock_conn)

    # Check that INSERT was called 3 times
    insert_calls = [call for call in mock_cursor.execute.mock_calls if 'INSERT INTO sales' in str(call)]
    assert len(insert_calls) == 3

    # Extract the parameters (second argument of execute)
    inserted_params = [call.args[1] for call in insert_calls]

    # Check that date_id values are correct (position 2 in the tuple)
    date_ids_used = [params[2] for params in inserted_params]
    assert date_ids_used == [20250101, 20250102, 20250103]

    # Bonus: check one full row
    assert inserted_params[0][:3] == ('C001', 'INV001', 20250101)
    assert inserted_params[1][:3] == ('C001', 'INV002', 20250102)
    assert inserted_params[2][:3] == ('C002', 'INV003', 20250103)


# =============================================
# Optional: Basic data schema test
# =============================================
def test_sample_data_schema(sample_df):
    required_cols = {
        'Client_ID', 'Invoice_Number', 'Date', 'Gross_Sales', 'Discount',
        'Net_Sales', 'CoG', 'VAT', 'Stamp_Duty', 'Total_Amount',
        'Client_Category', 'Payment_Method', 'Is_Return', 'Client_Type'
    }
    assert set(sample_df.columns) == required_cols
    assert pd.api.types.is_datetime64_any_dtype(sample_df['Date'])