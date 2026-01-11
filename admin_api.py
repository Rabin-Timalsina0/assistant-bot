import os
import uuid
import boto3
import requests
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.database import get_conn, release_conn
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType, FilterSelector
from app import database


def fetch_all(query: str, params: tuple = ()):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(query, params)
    columns = [c[0] for c in cur.description]
    rows = [dict(zip(columns, r)) for r in cur.fetchall()]
    cur.close()
    return rows

def fetch_one(query: str, params: tuple = ()):
    rows = fetch_all(query, params)
    return rows[0] if rows else None

def execute(query: str, params: tuple = ()):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        count = cur.rowcount
        cur.close()
        return count
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_conn(conn)

class Product(BaseModel):
    product_id: int
    product_name: str
    color: Optional[str] = None
    price: float
    stock_quantity: int
    description: Optional[str] = None
    image_urls: List[str] = []
    sizes: List[str] = []

class ProductCreate(BaseModel):
    product_name: str
    color: str
    price: float
    stock_quantity: int
    description: Optional[str] = None
    image_urls: Optional[List[str]] = []
    sizes: Optional[List[str]] = []

class ProductUpdate(BaseModel):
    product_name: Optional[str] = None
    color: Optional[str] = None
    price: Optional[float] = None
    stock_quantity: Optional[int] = None
    description: Optional[str] = None
    image_urls: Optional[List[str]] = None
    sizes: Optional[List[str]] = None

class OrderItem(BaseModel):
    product_id: int
    product_name: str
    color: Optional[str] = None
    size: Optional[str] = None
    quantity: int
    total_price: float

class Order(BaseModel):
    order_id: int
    customer_name: Optional[str] = None
    phone_number: Optional[str] = None
    shipping_address: Optional[str] = None
    status: str
    order_date: Optional[datetime] = None
    items: List[OrderItem]
    item_count: int
    order_total: float

class StatusUpdate(BaseModel):
    status: str

def extract_r2_key_from_url(url: str) -> str:
    """
    Extract the R2 object key from a URL
    """
    # Handle both custom domain and R2.dev URLs
    public_base = os.getenv("R2_PUBLIC_BASE")
    account_id = os.getenv("R2_ACCOUNT_ID")
    bucket = os.getenv("R2_BUCKET")
    
    if public_base and url.startswith(public_base):
        return url[len(public_base.rstrip('/') + '/'):]
    elif account_id and url.startswith(f"https://{account_id}.r2.dev/{bucket}/"):
        return url[len(f"https://{account_id}.r2.dev/{bucket}/"):]
    else:
        raise ValueError(f"Unable to extract key from URL: {url}")
    
def delete_from_r2(key: str):
    """
    Delete an object from R2 by its key
    """
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET")
    
    if not all([account_id, access_key, secret_key, bucket]):
        raise HTTPException(status_code=500, detail="R2 credentials not configured")
    
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )
        
        s3.delete_object(Bucket=bucket, Key=key)
        print(f"✅ Successfully deleted object from R2: {key}")
    except Exception as e:
        print(f"⚠️ Failed to delete object from R2: {e}")
        # Don't raise an exception here as it might break the main operation

def add_to_qdrant(product_id: int, name: str, color: str, image_urls: List[str]):
    if not image_urls:
        print(f"⚠️ No images for product {product_id}, skipping Qdrant addition")
        return

    points = []
    combined_name = f"{name}_{color}"

    for idx, url in enumerate(image_urls):
        try:
            # Embedding logic removed (like chatbot_NoImage)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=[],  # Empty vector as embedding is removed
                payload={"name": combined_name, "source_url": url}
            )
            points.append(point)
            print(f"✅ Successfully added image {idx} for product {product_id}")
        except Exception as e:
            print(f"⚠️ Failed to add image for {url}: {e}")

    if points:
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"✅ Successfully added {len(points)} vectors to Qdrant for product {product_id}")
        except Exception as e:
            print(f"⚠️ Failed to upsert points to Qdrant for product {product_id}: {e}")
    else:
        print(f"⚠️ No valid points to add to Qdrant for product {product_id}")

def remove_from_qdrant(name: str, color: str):
    try:
        combined_name = f"{name}_{color}"
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="name",
                            match=MatchValue(value=combined_name)
                        )
                    ]
                )
            )
        )
        print(f"Product {combined_name} removed successfully from Qdrant.")
    except Exception as e:
        print(f"⚠️ Failed to remove product {combined_name} from Qdrant: {e}")

@router.get("/stats")
def get_stats(client_id: str):
    total_orders = fetch_one("SELECT COUNT(*) AS c FROM orders WHERE client = %s", (client_id,))['c']
    total_products = fetch_one("SELECT COUNT(*) AS c FROM products WHERE client = %s", (client_id,))['c']
    total_images = fetch_one(
        """
        SELECT COUNT(*) AS c
        FROM product_images pi
        JOIN products p ON p.product_id = pi.product_id
        WHERE p.client = %s
        """,
        (client_id,)
    )['c']
    return {"total_orders": total_orders, "total_products": total_products, "total_images": total_images}

@router.get("/products", response_model=List[Product])
def get_products(client_id: str, q: Optional[str] = None):
    clauses = []
    params: List[str] = []
    clauses.append("p.client = %s"); params.append(client_id)
    
    if q:
        clauses.append("(p.product_name ILIKE %s OR p.color ILIKE %s)")
        params.extend([f"%{q}%", f"%{q}%"])
    
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    
    # Optimized query - single query with proper joins
    rows = fetch_all(
        f"""
        SELECT 
            p.product_id, 
            p.product_name, 
            p.color, 
            p.price, 
            p.stock_quantity, 
            p.description,
            COALESCE(
                ARRAY_AGG(DISTINCT pi.image_url) FILTER (WHERE pi.image_url IS NOT NULL),
                '{{}}'
            ) AS image_urls,
            COALESCE(
                ARRAY_AGG(DISTINCT ps.sizes) FILTER (WHERE ps.sizes IS NOT NULL),
                '{{}}'
            ) AS sizes
        FROM products p
        LEFT JOIN product_images pi ON p.product_id = pi.product_id AND pi.client = p.client
        LEFT JOIN product_sizes ps ON p.product_name = ps.product_name AND ps.client = p.client
        {where}
        GROUP BY p.product_id, p.product_name, p.color, p.price, p.stock_quantity, p.description
        ORDER BY p.product_id DESC
        """,
        tuple(params)
    )
    
    # Convert database arrays to Python lists
    for r in rows:
        if not isinstance(r['image_urls'], list):
            r['image_urls'] = list(r['image_urls']) if r['image_urls'] else []
        if not isinstance(r['sizes'], list):
            r['sizes'] = list(r['sizes']) if r['sizes'] else []
    
    return rows

@router.post("/products", response_model=Product)
def create_product(client_id: str, product: ProductCreate):
    conn = get_conn()
    try:
        cur = conn.cursor()

        # Check if product with same name and color already exists
        cur.execute(
            "SELECT product_id FROM products WHERE product_name = %s AND color = %s AND client = %s",
            (product.product_name, product.color, client_id)
        )
        existing = cur.fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Product with same name and color already exists")

        # Insert product
        cur.execute(
            """
            INSERT INTO products (product_name, color, price, stock_quantity, description, client)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING product_id
            """,
            (product.product_name, product.color, product.price, product.stock_quantity, product.description, client_id)
        )
        
        product_id = cur.fetchone()[0]

        # Insert images
        if product.image_urls:
            for url in product.image_urls:
                cur.execute(
                    "INSERT INTO product_images (product_id, image_url, client) VALUES (%s, %s, %s)", 
                    (product_id, url, client_id)
                )

        # Insert sizes using product_name
        if product.sizes:
            # Check if sizes already exist for this product_name
            cur.execute(
                "SELECT COUNT(*) as count FROM product_sizes WHERE product_name = %s AND client = %s",
                (product.product_name, client_id)
            )
            existing_sizes = cur.fetchone()[0]
            if existing_sizes == 0:
                for size in product.sizes:
                    cur.execute(
                        "INSERT INTO product_sizes (product_name, sizes, client) VALUES (%s, %s, %s)", 
                        (product.product_name, size, client_id)
                    )

        # Fetch created product
        cur.execute(
            """
            SELECT 
                p.product_id, 
                p.product_name, 
                p.color, 
                p.price, 
                p.stock_quantity, 
                p.description,
                COALESCE(
                    ARRAY_AGG(DISTINCT pi.image_url) FILTER (WHERE pi.image_url IS NOT NULL),
                    '{}'
                ) AS image_urls,
                COALESCE(
                    ARRAY_AGG(DISTINCT ps.sizes) FILTER (WHERE ps.sizes IS NOT NULL),
                    '{}'
                ) AS sizes
            FROM products p
            LEFT JOIN product_images pi ON p.product_id = pi.product_id AND pi.client = p.client
            LEFT JOIN product_sizes ps ON p.product_name = ps.product_name AND ps.client = p.client
            WHERE p.product_id = %s AND p.client = %s
            GROUP BY p.product_id, p.product_name, p.color, p.price, p.stock_quantity, p.description
            """,
            (product_id, client_id)
        )
        
        columns = [desc[0] for desc in cur.description]
        created = dict(zip(columns, cur.fetchone()))
        
        # Convert database arrays to Python lists
        if not isinstance(created['image_urls'], list):
            created['image_urls'] = list(created['image_urls']) if created['image_urls'] else []
        if not isinstance(created['sizes'], list):
            created['sizes'] = list(created['sizes']) if created['sizes'] else []

        # Commit the transaction
        conn.commit()

        # Add to Qdrant
        add_to_qdrant(product_id, created['product_name'], created['color'], created['image_urls'])

        return created
        
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        print(f"Error creating product: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        release_conn(conn)
    
@router.put("/products/{product_id}", response_model=Product)
def update_product(client_id: str, product_id: int, payload: ProductUpdate):
    conn = get_conn()
    try:
        cur = conn.cursor()

        # Get current product data before update
        cur.execute(
            "SELECT product_name, color FROM products WHERE product_id = %s AND client = %s",
            (product_id, client_id)
        )
        current_product = cur.fetchone()
        if not current_product:
            raise HTTPException(status_code=404, detail="Product not found")

        old_product_name = current_product[0]
        old_color = current_product[1]

        # Check if the update would create a duplicate
        if payload.product_name is not None or payload.color is not None:
            new_product_name = payload.product_name if payload.product_name is not None else old_product_name
            new_color = payload.color if payload.color is not None else old_color
            
            cur.execute(
                "SELECT product_id FROM products WHERE product_name = %s AND color = %s AND client = %s AND product_id != %s",
                (new_product_name, new_color, client_id, product_id)
            )
            duplicate = cur.fetchone()
            if duplicate:
                raise HTTPException(status_code=400, detail="Another product with same name and color already exists")

        # Update product fields
        updates = []
        params = []
        if payload.product_name is not None:
            updates.append("product_name = %s")
            params.append(payload.product_name)
        if payload.color is not None:
            updates.append("color = %s")
            params.append(payload.color)
        if payload.price is not None:
            updates.append("price = %s")
            params.append(payload.price)
        if payload.stock_quantity is not None:
            updates.append("stock_quantity = %s")
            params.append(payload.stock_quantity)
        if payload.description is not None:
            updates.append("description = %s")
            params.append(payload.description)
        
        if updates:
            params.extend([product_id, client_id])
            cur.execute(
                "UPDATE products SET " + ", ".join(updates) + " WHERE product_id = %s AND client = %s", 
                tuple(params)
            )

        # Handle images update
        if payload.image_urls is not None:
            cur.execute(
                "SELECT image_url FROM product_images WHERE product_id = %s AND client = %s",
                (product_id, client_id)
            )
            current_images = cur.fetchall()
            current_image_urls = [img[0] for img in current_images]
            
            # Find images to delete from R2
            images_to_remove = set(current_image_urls) - set(payload.image_urls)
            for url in images_to_remove:
                try:
                    key = extract_r2_key_from_url(url)
                    delete_from_r2(key)
                except Exception as e:
                    print(f"⚠️ Failed to delete image from R2: {e}")
            
            # Update database
            cur.execute("DELETE FROM product_images WHERE product_id = %s AND client = %s", (product_id, client_id))
            for url in payload.image_urls:
                cur.execute(
                    "INSERT INTO product_images (product_id, image_url, client) VALUES (%s, %s, %s)", 
                    (product_id, url, client_id)
                )

        # Update sizes
        if payload.sizes is not None:
            target_product_name = payload.product_name if payload.product_name is not None else old_product_name
            
            cur.execute("DELETE FROM product_sizes WHERE product_name = %s AND client = %s", (target_product_name, client_id))
            
            for size in payload.sizes:
                cur.execute(
                    "INSERT INTO product_sizes (product_name, sizes, client) VALUES (%s, %s, %s)", 
                    (target_product_name, size, client_id)
                )

        # Fetch updated product
        cur.execute(
            """
            SELECT 
                p.product_id, 
                p.product_name, 
                p.color, 
                p.price, 
                p.stock_quantity, 
                p.description,
                COALESCE(
                    ARRAY_AGG(DISTINCT pi.image_url) FILTER (WHERE pi.image_url IS NOT NULL),
                    '{}'
                ) AS image_urls,
                COALESCE(
                    ARRAY_AGG(DISTINCT ps.sizes) FILTER (WHERE ps.sizes IS NOT NULL),
                    '{}'
                ) AS sizes
            FROM products p
            LEFT JOIN product_images pi ON p.product_id = pi.product_id AND pi.client = p.client
            LEFT JOIN product_sizes ps ON p.product_name = ps.product_name AND ps.client = p.client
            WHERE p.product_id = %s AND p.client = %s
            GROUP BY p.product_id, p.product_name, p.color, p.price, p.stock_quantity, p.description
            """,
            (product_id, client_id)
        )
        
        columns = [desc[0] for desc in cur.description]
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Product not found")
            
        result = dict(zip(columns, row))
        
        if not isinstance(result['image_urls'], list):
            result['image_urls'] = list(result['image_urls']) if result['image_urls'] else []
        if not isinstance(result['sizes'], list):
            result['sizes'] = list(result['sizes']) if result['sizes'] else []

        # Commit transaction
        conn.commit()

        # Update Qdrant
        remove_from_qdrant(old_product_name, old_color)
        add_to_qdrant(product_id, result['product_name'], result['color'], result['image_urls'])

        return result
        
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        print(f"Error updating product: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        release_conn(conn)

@router.delete("/products/{product_id}")
def delete_product(client_id: str, product_id: int):
    try:
        # Get product data before deletion
        product = fetch_one(
            "SELECT product_name, color FROM products WHERE product_id = %s AND client = %s",
            (product_id, client_id)
        )
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
            
        product_images = fetch_all(
            "SELECT image_url FROM product_images WHERE product_id = %s",
            (product_id,)
        )
        
        # Delete the product
        count = execute("DELETE FROM products WHERE product_id = %s AND client = %s", (product_id, client_id))
        if count == 0:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Delete images from R2
        for image in product_images:
            try:
                key = extract_r2_key_from_url(image['image_url'])
                delete_from_r2(key)
            except Exception as e:
                print(f"⚠️ Failed to delete image from R2: {e}")
        
        # FIXED: Check if there are other products with the same product_name before deleting sizes
        other_products = fetch_one(
            "SELECT COUNT(*) as count FROM products WHERE product_name = %s AND client = %s",
            (product['product_name'], client_id)
        )
        
        # Only delete sizes if this was the last product with this name
        if other_products['count'] == 0:
            execute("DELETE FROM product_sizes WHERE product_name = %s AND client = %s", (product['product_name'], client_id))
        
        # Remove from Qdrant
        remove_from_qdrant(product['product_name'], product['color'])
        
        return {"message": "Product deleted from DB, R2, and Qdrant"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting product: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@router.get("/revenue")
def get_revenue(client_id: str):
    total_revenue = fetch_one(
        "SELECT COALESCE(SUM(total_price), 0) AS total_revenue FROM orders WHERE client = %s AND status = 'delivered'",
        (client_id,)
    )['total_revenue']
    return {"total_revenue": total_revenue}

@router.get("/orders", response_model=List[Order])
def get_orders(client_id: str, status: Optional[str] = None, q: Optional[str] = None):
    clauses = []
    params: List[object] = []
    clauses.append("o.client = %s"); params.append(client_id)
    if status:
        clauses.append("LOWER(o.status) = LOWER(%s)"); params.append(status)
    if q:
        clauses.append("(CAST(o.order_id AS TEXT) ILIKE %s OR o.customer_name ILIKE %s OR o.phone_number ILIKE %s OR o.product_name ILIKE %s)")
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%"])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    
    # Group by order_id to get order-level information
    rows = fetch_all(
        f"""
        SELECT 
            o.order_id,
            o.customer_name,
            o.phone_number,
            o.shipping_address,
            o.status,
            MAX(o.order_date) as order_date,  -- Use MAX to get the most recent order date
            -- Aggregate products into an array
            json_agg(
                json_build_object(
                    'product_id', o.product_id,
                    'product_name', o.product_name,
                    'color', o.color,
                    'size', o.size,
                    'quantity', o.quantity,
                    'total_price', o.total_price
                ) ORDER BY o.product_id
            ) as items,
            COUNT(o.product_id) as item_count,
            SUM(o.total_price) as order_total
        FROM orders o
        {where}
        GROUP BY o.order_id, o.customer_name, o.phone_number, o.shipping_address, o.status
        ORDER BY MAX(o.order_date) DESC
        """,
        tuple(params)
    )
    return rows

@router.put("/orders/{order_id}/status")
def set_order_status(client_id: str, order_id: int, body: StatusUpdate):
    if body.status.lower() not in ("pending", "shipped", "delivered"):
        raise HTTPException(status_code=400, detail="Invalid status")
    count = execute("UPDATE orders SET status = %s WHERE order_id = %s AND client = %s", (body.status, order_id, client_id))
    if count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"order_id": order_id, "status": body.status}

@router.delete("/orders/{order_id}")
def delete_order(client_id: str, order_id: int):
    count = execute("DELETE FROM orders WHERE order_id = %s AND client = %s", (order_id, client_id))
    if count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"message": f"Order {order_id} deleted"}

@router.post("/upload/image")
async def upload_image(client_id: str, file: UploadFile = File(...)):
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET")
    public_base = os.getenv("R2_PUBLIC_BASE", "https://pub-0fa1e47317994f149ac6ccc049b7957c.r2.dev")

    if not all([account_id, access_key, secret_key, bucket]):
        raise HTTPException(status_code=500, detail="R2 credentials not configured")

    ext = os.path.splitext(file.filename)[1] or ".jpg"
    key = f"{client_id}/{uuid.uuid4().hex}{ext}"

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

        content = await file.read()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,
            ContentType=file.content_type or "application/octet-stream",
        )

        if public_base:
            url = f"{public_base.rstrip('/')}/{key}"
        else:
            url = f"https://{account_id}.r2.dev/{bucket}/{key}"

        return {"url": url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {e}")

@router.post("/upload/get_url")
def get_r2_upload_url(client_id: str, payload: dict):
    filename = payload.get("filename")
    content_type = payload.get("content_type", "application/octet-stream")
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")

    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET")
    public_base = os.getenv("R2_PUBLIC_BASE")
    if not all([account_id, access_key, secret_key, bucket]):
        raise HTTPException(status_code=500, detail="R2 credentials not configured")

    key = f"{client_id}/{filename}"
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto'
        )
        upload_url = s3.generate_presigned_url(
            'put_object',
            Params={'Bucket': bucket, 'Key': key, 'ContentType': content_type},
            ExpiresIn=600,
            HttpMethod='PUT'
        )
        public_url = f"{public_base.rstrip('/')}/{key}" if public_base else f"https://{account_id}.r2.dev/{bucket}/{key}"
        return {"upload_url": upload_url, "public_url": public_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create signed URL: {e}")

@router.post("/products/add_image")
def add_product_image(client_id: str, payload: dict):
    product_id = payload.get("product_id")
    image_url = payload.get("image_url")
    if not product_id or not image_url:
        raise HTTPException(status_code=400, detail="product_id and image_url required")
    owner = fetch_one("SELECT 1 FROM products WHERE product_id = %s AND client = %s", (product_id, client_id))
    if not owner:
        raise HTTPException(status_code=404, detail="Product not found")
    execute("INSERT INTO product_images (product_id, image_url) VALUES (%s, %s)", (product_id, image_url))
    return {"ok": True}

@router.post("/products/remove_image")
def remove_product_image(client_id: str, payload: dict):
    product_id = payload.get("product_id")
    image_url = payload.get("image_url")
    if not product_id or not image_url:
        raise HTTPException(status_code=400, detail="product_id and image_url required")
    
    owner = fetch_one("SELECT 1 FROM products WHERE product_id = %s AND client = %s", (product_id, client_id))
    if not owner:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Delete from database
    execute("DELETE FROM product_images WHERE product_id = %s AND image_url = %s", (product_id, image_url))
    
    # Delete from R2
    try:
        key = extract_r2_key_from_url(image_url)
        delete_from_r2(key)
    except Exception as e:
        print(f"⚠️ Failed to delete image from R2: {e}")
        # You might want to raise an exception here or handle it differently
    
    return {"ok": True}
