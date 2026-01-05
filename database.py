import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
# Use environment variable in production - no hardcoded fallback for security
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL environment variable must be set")

# Initialize a connection pool (min=1, max=10 connections)
db_pool = pool.SimpleConnectionPool(1, 10, DB_URL)

def get_conn():
    """Get a connection from the pool."""
    return db_pool.getconn()

def release_conn(conn):
    """Release the connection back to the pool."""
    db_pool.putconn(conn)


# ---------------- Your functions ---------------- #

def get_clients(client):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        query = 'SELECT * FROM clients where name = %s;'
        cursor.execute(query, (client,))
        response = cursor.fetchall()
        cursor.close()
        return [item for item in response[0]]
    finally:
        release_conn(conn)


def get_menu(client):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        query = '''
            SELECT product_name, price, SUM(stock_quantity) AS total_stock
            FROM products
            WHERE client = %s
            GROUP BY product_name, price
            ORDER BY product_name;
        '''
        cursor.execute(query, (client,))
        products = cursor.fetchall()

        query_colors = '''
            SELECT product_name, color, stock_quantity
            FROM products
            WHERE client = %s
            ORDER BY product_name, color;
        '''
        cursor.execute(query_colors, (client,))
        colors_data = cursor.fetchall()

        colors_by_product = {}
        for pname, color, qty in colors_data:
            colors_by_product.setdefault(pname, []).append((color, qty))

        result = []
        for pname, price, total_stock in products:
            stock_status = "In stock" if total_stock > 0 else "Out of stock"
            color_str = ", ".join(
                f"{col}({'In stock' if qty > 0 else 'Out of stock'})"
                for col, qty in colors_by_product.get(pname, [])
            )
            result.append((pname, price, stock_status, color_str))

        cursor.close()
        return result
    finally:
        release_conn(conn)


def get_names(client):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        query = '''
            SELECT DISTINCT product_name
            FROM products
            WHERE client = %s
        '''
        cursor.execute(query, (client,))
        response = cursor.fetchall()
        cursor.close()
        return [item[0] for item in response]
    finally:
        release_conn(conn)


def get_colors(client, product):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        query = '''
            SELECT color
            FROM products
            WHERE product_name = %s AND client = %s
        '''
        cursor.execute(query, (product, client))
        response = cursor.fetchall()
        cursor.close()
        return [item[0] for item in response]
    finally:
        release_conn(conn)

def get_sizes(client, product):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        query = '''
            SELECT DISTINCT sizes
            FROM product_sizes
            WHERE product_name = %s AND client = %s
        '''
        cursor.execute(query, (product, client))
        response = cursor.fetchall()
        cursor.close()
        return [item[0] for item in response if item[0] is not None]
    finally:
        release_conn(conn)


def get_desc(client, item):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        query = '''
            SELECT product_name, description, price, stock_quantity, color
            FROM products
            WHERE product_name = %s AND client = %s
        '''
        cursor.execute(query, (item, client))
        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            return f"No product found for '{item}'."

        product_name, description, price, _, _ = rows[0]
        colors_stock = ", ".join(
            f"{row[4]}({'In Stock' if row[3] > 0 else 'Out Stock'})"
            for row in rows
        )
        final = f"{product_name}: Rs.{price:.2f} | {description} | Colors: {colors_stock}"
        return final
    finally:
        release_conn(conn)


def get_order_id(client):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        query = 'SELECT MAX(order_id) FROM orders WHERE client = %s'
        cursor.execute(query, (client,))
        result = cursor.fetchone()[0]
        cursor.close()
        return 1 if result is None else result + 1
    finally:
        release_conn(conn)


def get_total_price(client, order_id):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        query = 'SELECT total_price FROM orders WHERE order_id = %s AND client = %s'
        cursor.execute(query, (order_id, client))
        rows = cursor.fetchall()
        cursor.close()
        return sum(row[0] for row in rows)
    finally:
        release_conn(conn)


def get_urls(client, product, color=None):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        if color is None:
            query = '''
            SELECT pi.image_url
            FROM products p
            JOIN product_images pi ON p.product_id = pi.product_id
            WHERE p.product_name = %s AND p.client = %s
            '''
            params = (product, client)
        else:
            query = '''
            SELECT pi.image_url
            FROM products p
            JOIN product_images pi ON p.product_id = pi.product_id
            WHERE p.product_name = %s AND p.color = %s AND p.client = %s
            '''
            params = (product, color, client)
            
        cursor.execute(query, params)
        image_urls = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return image_urls
    finally:
        release_conn(conn)


def insert_order_item(client, order_id, product_name, quantity, color, size, status, name, phone, address):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT product_id FROM products
            WHERE product_name = %s AND color = %s AND client = %s
        ''', (product_name, color, client))
        product_id = cursor.fetchone()
        if not product_id:
            return -1
        product_id = product_id[0]

        cursor.execute('SELECT price FROM products WHERE product_id = %s', (product_id,))
        price = cursor.fetchone()[0]
        total_price = price * quantity

        cursor.execute('''
            INSERT INTO orders (order_id, product_id, product_name, color, size, quantity, total_price, status, customer_name, phone_number, shipping_address, client)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ''', (order_id, product_id, product_name, color, size, quantity, total_price, status, name, phone, address, client))

        conn.commit()
        cursor.close()
        return 1
    except Exception as e:
        print(f"Error inserting order: {e}")
        conn.rollback()
        return -1
    finally:
        release_conn(conn)


def get_status(client, order_id):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        query = "SELECT status FROM orders WHERE order_id = %s AND client = %s"
        cursor.execute(query, (order_id, client))
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else None
    finally:
        release_conn(conn)


def cancel_order(client, order_id):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE orders
            SET status = 'Canceled'
            WHERE order_id = %s AND client = %s
        """, (order_id, client))
        conn.commit()
        rowcount = cursor.rowcount
        cursor.close()
        return 1 if rowcount > 0 else -1
    except Exception as e:
        print(f"Error canceling order: {e}")
        conn.rollback()
        return -2
    finally:
        release_conn(conn)
