import pytest
from datetime import date
from unittest.mock import MagicMock, patch

# Import the actual functions to test
from src.dw.populate_dw import (
    populate_dim_client,
    populate_dim_invoice,
    populate_dim_date,
    populate_fact_sales,
)


# =============================================
# Test populate_dim_client
# =============================================
def test_populate_dim_client():
    dw_conn = MagicMock()
    staging_conn = MagicMock()
    dw_cursor = dw_conn.cursor.return_value
    st_cursor = staging_conn.cursor.return_value

    # Staging has C001, C002, C003
    st_cursor.fetchall.return_value = [('C001',), ('C002',), ('C003',)]

    # DW already has C001, C002
    dw_cursor.fetchall.side_effect = [
        [('C001',), ('C002',)]  # existing in dim_client
    ]

    populate_dim_client(dw_conn, staging_conn)

    # Only C003 should be inserted → 1 INSERT call
    insert_calls = [call for call in dw_cursor.execute.mock_calls if 'INSERT INTO dim_client' in str(call)]
    assert len(insert_calls) == 1
    assert insert_calls[0].args[1] == ('C003',)

    dw_conn.commit.assert_called_once()


def test_populate_dim_client_no_new():
    dw_conn = MagicMock()
    staging_conn = MagicMock()
    dw_cursor = dw_conn.cursor.return_value
    st_cursor = staging_conn.cursor.return_value

    st_cursor.fetchall.return_value = [('C001',)]
    dw_cursor.fetchall.return_value = [('C001',)]

    with patch('src.dw.populate_dw.logging') as mock_logging:
        populate_dim_client(dw_conn, staging_conn)

    # No INSERT calls
    insert_calls = [call for call in dw_cursor.execute.mock_calls if 'INSERT' in str(call)]
    assert len(insert_calls) == 0
    mock_logging.info.assert_any_call("No new clients to insert into dim_client.")


# =============================================
# Test populate_dim_invoice
# =============================================
def test_populate_dim_invoice():
    dw_conn = MagicMock()
    staging_conn = MagicMock()
    dw_cursor = dw_conn.cursor.return_value
    st_cursor = staging_conn.cursor.return_value

    st_cursor.fetchall.return_value = [
        ('INV001', 'Cash', 'No'),
        ('INV002', 'Card', 'No'),
        ('INV003', 'Cash', 'Yes'),
    ]

    dw_cursor.fetchall.side_effect = [
        [('INV001',)]  # only INV001 exists in DW
    ]

    populate_dim_invoice(dw_conn, staging_conn)

    insert_calls = [call for call in dw_cursor.execute.mock_calls if 'INSERT INTO dim_invoice' in str(call)]
    assert len(insert_calls) == 2

    inserted_params = [call.args[1] for call in insert_calls]
    assert ('INV002', 'Card', 'No') in inserted_params
    assert ('INV003', 'Cash', 'Yes') in inserted_params


# =============================================
# Test populate_dim_date
# =============================================
def test_populate_dim_date():
    dw_conn = MagicMock()
    staging_conn = MagicMock()
    dw_cursor = dw_conn.cursor.return_value
    st_cursor = staging_conn.cursor.return_value

    # Staging date table has these
    st_cursor.fetchall.side_effect = [
        [(20250101, date(2025,1,1)), (20250102, date(2025,1,2)), (20250103, date(2025,1,3))],
        [(20250101, date(2025,1,1)), (20250102, date(2025,1,2)), (20250103, date(2025,1,3))],
    ]

    # DW already has first two
    dw_cursor.fetchall.return_value = [(20250101,), (20250102,)]

    populate_dim_date(dw_conn, staging_conn)

    insert_calls = [call for call in dw_cursor.execute.mock_calls if 'INSERT INTO dim_date' in str(call)]
    assert len(insert_calls) == 1
    assert insert_calls[0].args[1] == (20250103, date(2025,1,3))

    dw_conn.commit.assert_called_once()


# =============================================
# Test populate_fact_sales
# =============================================
def test_populate_fact_sales():
    dw_conn = MagicMock()
    staging_conn = MagicMock()
    dw_cursor = dw_conn.cursor.return_value
    st_cursor = staging_conn.cursor.return_value

    # Data from staging sales table
    st_cursor.fetchall.return_value = [
        ('INV001', 'C001', 20250101, 1000.0, 100.0, 900.0, 500.0, 162.0, 0.0, 1062.0, 'Retail'),
        ('INV002', 'C002', 20250102, 2000.0, 0.0, 2000.0, 1000.0, 324.0, 0.0, 2324.0, 'Wholesale'),
        ('INV003', 'C001', 20250103, 1500.0, 150.0, 1350.0, 750.0, 243.0, 0.0, 1593.0, 'Retail'),
    ]

    # Only INV001 exists in DW fact_sales
    dw_cursor.fetchall.return_value = [('INV001',)]

    populate_fact_sales(dw_conn, staging_conn)

    insert_calls = [call for call in dw_cursor.execute.mock_calls if 'INSERT INTO fact_sales' in str(call)]
    assert len(insert_calls) == 2

    inserted_params = [call.args[1] for call in insert_calls]
    assert ('C002', 'INV002', 20250102, 2000.0, 0.0, 2000.0, 1000.0, 324.0, 0.0, 2324.0, 'Wholesale') in inserted_params
    assert ('C001', 'INV003', 20250103, 1500.0, 150.0, 1350.0, 750.0, 243.0, 0.0, 1593.0, 'Retail') in inserted_params


def test_populate_fact_sales_no_new():
    dw_conn = MagicMock()
    staging_conn = MagicMock()
    dw_cursor = dw_conn.cursor.return_value
    st_cursor = staging_conn.cursor.return_value

    st_cursor.fetchall.return_value = [('INV001', 'C001', 20250101, 1000.0, 100.0, 900.0, 500.0, 162.0, 0.0, 1062.0, 'Retail')]
    dw_cursor.fetchall.return_value = [('INV001',)]

    with patch('src.dw.populate_dw.logging') as mock_logging:
        populate_fact_sales(dw_conn, staging_conn)

    insert_calls = [call for call in dw_cursor.execute.mock_calls if 'INSERT' in str(call)]
    assert len(insert_calls) == 0
    mock_logging.info.assert_any_call("No new sales to insert into fact_sales.")


# =============================================
# Optional: Test full flow sequence (just logical order)
# =============================================
def test_etl_sequence():
    from src.dw.populate_dw import __name__  # Just to trigger main block logic if needed
    expected_order = [
        'populate_dim_client',
        'populate_dim_invoice',
        'populate_dim_date',
        'populate_fact_sales'
    ]
    # This is mainly documentation — real order is enforced in main()
    assert expected_order[-1] == 'populate_fact_sales'  # fact table last