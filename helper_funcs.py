import requests
from difflib import get_close_matches
from app import database, state_manager
from collections import deque, defaultdict
import tiktoken
import re
import spacy
from app.rag import FAQRetriever

nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words
USER_ID = "598966519963758"

def remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if token.is_alpha and token.text.lower() not in stopwords])

token_encoder = tiktoken.get_encoding("cl100k_base")

def extract_order_id(text):
    """Extracts order ID from text with context awareness"""
    # Look for patterns that explicitly mention order ID
    patterns = [
        r'order\s+#?(\d+)',           # "order #123" or "order 123"
        r'order\s+(\d+)',             # "order 123"
        r'track\s+order\s+#?(\d+)',   # "track order #123"
        r'track\s+order\s+(\d+)',     # "track order 123"
        r'order\s+id\s+#?(\d+)',      # "order id #123"
        r'order\s+id\s+(\d+)',        # "order id 123"
        r'#(\d+)',                    # "#123" (standalone)
        r'my\s+order\s+is\s+(\d+)',   # "my order is 123"
        r'id\s+is\s+(\d+)',           # "id is 123"
        r'(\d+)\s*$',                 # "123" at end of sentence
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # If no explicit order ID pattern found, return None
    return None


def count_tokens(text):
    return len(token_encoder.encode(text))

def build_alias_map():
    mapping = {
        "extra extra small": "XXS", "xxs": "XXS", "extra small": "XS", "x-small": "XS", "xs": "XS",
        "small": "S", "s": "S",
        "medium": "M", "med": "M", "m": "M",
        "large": "L", "l": "L",
        "extra large": "XL", "x-large": "XL", "xl": "XL",
        "2xl": "XXL", "xxl": "XXL", "extra extra large": "XXL",
        "3xl": "3XL", "xxxL": "3XL", "xxxl": "3XL",
    }
    return mapping

def extract_size(user_input, valid_sizes, extra_aliases=None):
    if not user_input or not user_input.strip():
        return None
    
    alias_map = build_alias_map()
    if extra_aliases:
        alias_map.update({k.lower(): v for k, v in extra_aliases.items()})
    
    normalized = re.sub(r"[^\w\s]", " ", user_input).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    aliases_sorted = sorted(alias_map.keys(), key=lambda x: -len(x))
    for alias in aliases_sorted:
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, normalized):
            canonical = alias_map[alias]
            if valid_sizes is None or canonical in valid_sizes:
                return canonical
    
    tokens = normalized.split()
    for tok in tokens:
        if tok in alias_map:
            canonical = alias_map[tok]
            if valid_sizes is None or canonical in valid_sizes:
                return canonical
        if "/" in tok or "," in tok:
            subtoks = re.split(r"[\/,]", tok)
            for st in subtoks:
                st = st.strip()
                if st in alias_map:
                    canonical = alias_map[st]
                    if valid_sizes is None or canonical in valid_sizes:
                        return canonical
    
    for code in (valid_sizes):
        pattern = r"\b" + re.escape(code.lower()) + r"\b"
        if re.search(pattern, normalized):
            return code
    
    return None

def send_message_facebook(recipient_id: str, message_data, image_urls=None, access_token=None):
    if not access_token:
        raise ValueError("Access token is required for sending messages")
        
    url = f"https://graph.facebook.com/v24.0/{USER_ID}/messages"
    params = {"access_token": access_token}
    headers = {"Content-Type": "application/json"}

    if isinstance(message_data, str):
        payload = {"text": message_data}
    else:
        payload = message_data

    data = {
        "recipient": {"id": recipient_id},
        "message": payload
    }

    response = requests.post(url, params=params, headers=headers, json=data)
    response_data = response.json()

    if response.status_code == 200:
        print("Message sent successfully:", response_data)
    else:
        print("Error sending message:", response_data)
        return False

    if image_urls and isinstance(image_urls, list):
        for img_url in image_urls:
            img_data = {
                "recipient": {"id": recipient_id},
                "message": {
                    "attachment": {
                        "type": "image",
                        "payload": {
                            "url": img_url,
                            "is_reusable": True
                        }
                    }
                }
            }
            img_response = requests.post(url, params=params, headers=headers, json=img_data)
            img_response_data = img_response.json()

            if img_response.status_code == 200:
                print(f"Image sent successfully: {img_url}")
            else:
                print(f"Error sending image {img_url}:", img_response_data)

    return True

def normalize_product_name(user_product, valid_products):
    user_input = user_product.lower().strip()
    
    for product in valid_products:
        if user_input == product.lower():
            return product
    
    for product in valid_products:
        if product.lower().startswith(user_input):
            return product
    
    full_names = [p.lower() for p in valid_products]
    matches = get_close_matches(user_input, full_names, n=1, cutoff=0.7)
    if matches:
        idx = full_names.index(matches[0])
        return valid_products[idx]
    
    for product in valid_products:
        if user_input in product.lower():
            return product
    
    return None

def extract_from_caption(caption, client_id):
    match = re.search(r'#product_(\w+)', caption)
    if match:
        output = match.group(1)
        return normalize_product_name(output, database.get_names(client_id))
    return None

def extract_color(user_input, valid_colors):
    stop_words = {"i", "would", "like", "it", "in", "the", "a", "an", "to", 
                  "for", "please", "want", "prefer", "color", "colour", "shade",
                  "is", "am", "are", "be", "that", "this", "one", "maybe", "probably"}
    
    words = [word.strip().lower() for word in re.split(r'\W+', user_input) if word.strip()]

    for word in words:
        if word in stop_words:
            continue
            
        color = normalize_product_name(word, valid_colors)
        if color:
            return color
    
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if words[i] in stop_words or words[i+1] in stop_words:
            continue
            
        color = normalize_product_name(phrase, valid_colors)
        if color:
            return color
    
    return normalize_product_name(user_input, valid_colors)

    response = retriever.get_answer(user_input)
    return response

def generate_resumption_message(client, pending_action, pending_data, user_question, bot_response):
    """
    Generate a natural follow-up message to resume the pending action after an interruption.
    
    Args:
        client: OpenAI client instance
        pending_action: The pending action type (need_color, need_size, need_customer_info)
        pending_data: Data associated with the pending action (product, color, etc.)
        user_question: The interrupting question from the user
        bot_response: The response to the interrupting question
    
    Returns:
        A natural, contextual resumption message (str)
    """
    # Build context for GPT based on pending action type
    context_info = ""
    
    if pending_action == "need_color":
        product = pending_data.get("product", "item")
        context_info = f"The user was in the process of ordering a {product}, and we were asking them to choose a color."
    
    elif pending_action == "need_size":
        product = pending_data.get("product", "item")
        color = pending_data.get("color", "")
        if color:
            context_info = f"The user was in the process of ordering a {color} {product}, and we were asking them to choose a size."
        else:
            context_info = f"The user was in the process of ordering a {product}, and we were asking them to choose a size."
    
    elif pending_action == "need_customer_info":
        field = pending_data.get("next_field", "information")
        context_info = f"The user was in the checkout process, and we were asking them to provide their {field}."
    
    else:
        # Generic fallback
        context_info = "The user was in the middle of placing an order."
    
    # Create prompt for GPT to generate a natural resumption message
    prompt = f"""You are a friendly e-commerce chatbot. The user interrupted the order process with a question, and you answered it.

Context: {context_info}

User's interrupting question: "{user_question}"
Your response: "{bot_response}"

Now generate a SHORT, FRIENDLY follow-up message (1-2 sentences max) to naturally transition back to completing their order. The message should:
- Acknowledge their question was answered
- Smoothly guide them back to the pending action
- Be conversational and varied (don't use the same phrasing every time)
- Use emojis sparingly if appropriate

Generate ONLY the follow-up message, nothing else:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates natural, conversational follow-up messages for an e-commerce chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Higher temperature for more variety
            max_tokens=100
        )
        
        resumption_msg = response.choices[0].message.content.strip()
        # Remove quotes if GPT added them
        if resumption_msg.startswith('"') and resumption_msg.endswith('"'):
            resumption_msg = resumption_msg[1:-1]
        if resumption_msg.startswith("'") and resumption_msg.endswith("'"):
            resumption_msg = resumption_msg[1:-1]
        
        return resumption_msg
    
    except Exception as e:
        print(f"Error generating resumption message: {e}")
        # Fallback to a simple generic message
        if pending_action == "need_color":
            return "Now, what color would you like?"
        elif pending_action == "need_size":
            return "What size would you prefer?"
        elif pending_action == "need_customer_info":
            field = pending_data.get("next_field", "information")
            return f"Could you please provide your {field} to complete the order?"
        else:
            return "Shall we continue with your order?"

def extract_and_update_context(client_id, sender_id, user_text):
    """
    Extracts product, color, and size from user text and updates the persistent context.
    This runs on every message to keep track of what the user is talking about.
    """
    valid_products = database.get_names(client_id)
    
    # 1. Try to find a product
    product = normalize_product_name(user_text, valid_products)
    
    updates = {}
    if product:
        updates["product"] = product
        # If a new product is mentioned, we might want to clear old color/size?
        # For now, let's assume if they switch products, they might want to keep size but maybe not color.
        # But to be safe and simple, let's just update the product.
        # Actually, if I say "I want the hoodie", and then "in red", it works.
        # If I say "I want the t-shirt", the context updates to t-shirt.
        # If I then say "blue", it should be blue t-shirt.
    
    # 2. Try to find color (using the product we just found OR the one in context)
    current_context = state_manager.get_product_context(client_id, sender_id)
    context_product = product if product else current_context.get("product")
    
    if context_product:
        valid_colors = database.get_colors(client_id, context_product)
        color = extract_color(user_text, valid_colors)
        if color:
            updates["color"] = color
            
        valid_sizes = database.get_sizes(client_id, context_product)
        size = extract_size(user_text, valid_sizes)
        if size:
            updates["size"] = size

    if updates:
        state_manager.update_product_context(client_id, sender_id, updates)
        print(f"Updated context for {sender_id}: {updates}")

def extract_customer_info(user_input, field_name):
    patterns = {
        "name": [
            r"my name is ([\w\s]+)",
            r"i am ([\w\s]+)",
            r"name:? ([\w\s]+)",
            r"it's ([\w\s]+)",
            r"its ([\w\s]+)",
            r"call me ([\w\s]+)"
        ],
        "phone": [
            r"phone:? (\d{10})",
            r"number:? (\d{10})",
            r"its (\d{10})",
            r"(\d{10})",
            r"contact:? (\d{10})"
        ],
        "address": [
            r"address:? (.+)",
            r"deliver to (.+)",
            r"shipping:? (.+)",
            r"location:? (.+)",
            r"its (.+)",
            r"send to (.+)"
        ]
    }
    
    # Try patterns for the specific field
    for pattern in patterns.get(field_name, []):
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return the entire input
    return user_input

def save_to_db(client_id, items, customer_info):
    new_order_id = database.get_order_id(client_id)
    status = "Pending"
    name = customer_info["name"]
    phone = customer_info["phone"]
    address = customer_info["address"]
    for item in items:
        product = item['name']
        quantity = item['quantity']
        color = item['color']
        size = item['size']
        rcode = database.insert_order_item(client_id, new_order_id, product, quantity, color, size, status, name, phone, address)
        if rcode == -1:
            print("Error inserting order item")
            return -1
    return new_order_id
    
def greeting():
    return "Hello, how can I help you"

def conversation_continuation():
    """Handle conversation continuation responses like thanks, okay, etc."""
    # Return empty string to avoid unnecessary responses
    # This keeps conversations concise when users just acknowledge
    return ""

def inventory_inquiry(client_id):
    rspn = "\n\n".join([
        f"{item[0]}: Rs.{item[1]:.2f} - {item[2]}\nAvailable colors: {item[3]}"
        for item in database.get_menu(client_id)
    ])
    return f"Here are the available items:\n\n{rspn}"

def handle_function_call(client_id, sender_id, name, args: dict):
    # Get client-specific valid products
    valid_products = database.get_names(client_id)
    
    if name == "item_inquiry":
        product_name = args.get("product_name")
        color_query = args.get("color")
        size_query = args.get("size")
        
        context = state_manager.get_conversation_context(client_id, sender_id)
        
        # If user previously sent image or bot asked for product
        if context.get("pending_action") in ["need_product", "need_command"]:
            product_name = context.get("pending_data", {}).get("product_name")
            # Clear the state after using it
            state_manager.clear_pending_action(client_id, sender_id)

        # Normalize product name
        product = normalize_product_name(product_name, valid_products)

        if not product:
            return {
                "status": "failure",
                "message": f"Sorry, we don't have {product_name}",
                "suggestions": valid_products
            }

        # Get persistent product context to check if we have more info
        product_context = state_manager.get_product_context(client_id, sender_id)
        
        # Collect all available product details (from args or context)
        potential_order_data = {
            "product": product,
            "quantity": 1
        }
        
        # Check color availability
        color = None
        if color_query:
            valid_colors = database.get_colors(client_id, product)
            color = normalize_product_name(color_query, valid_colors)
            if not color:
                available = ", ".join(valid_colors) if valid_colors else "none"
                return {
                    "status": "success",
                    "message": f"Sorry, {product} is not available in {color_query}. Available colors: {available}"
                }
        elif product_context.get("color"):
            # Use color from persistent context
            color = product_context.get("color")
        
        if color:
            potential_order_data["color"] = color
        
        # Check size availability
        size = None
        if size_query:
            valid_sizes = database.get_sizes(client_id, product)
            size = extract_size(size_query, valid_sizes)
            if not size:
                available = ", ".join(valid_sizes) if valid_sizes else "none"
                return {
                    "status": "success",
                    "message": f"Sorry, {product} is not available in size {size_query}. Available sizes: {available}"
                }
        elif product_context.get("size"):
            # Use size from persistent context
            size = product_context.get("size")
        
        if size:
            potential_order_data["size"] = size
        
        # If we have color and/or size, set up a potential order
        if color or size:
            state_manager.set_pending_action(client_id, sender_id, "potential_order", potential_order_data)
            
            # Build response message
            details = []
            if color:
                details.append(f"in {color}")
            if size:
                details.append(f"size {size}")
            
            detail_str = " ".join(details)
            return {
                "status": "success",
                "message": f"The {product} {detail_str} is available. Would you like to place an order?"
            }

        # General product inquiry (no color or size specified)
        return {
            "status": "success",
            "details": database.get_desc(client_id, product)
        }

    
    elif name == "add_item":
        response = {
            "status": "fail",
            "product": None,
            "color": None,
            "size": None,
            "quantity": 1,
            "message": "",
            "cart": [],
            "next_step": None
        }
        
        product = None
        color = None
        size = None
        quantity = 1
        
        context = state_manager.get_conversation_context(client_id, sender_id)
        pending_action = context.get("pending_action")
        pending_data = context.get("pending_data", {})

        # First, try to get color and size from args (they might be provided directly)
        color_from_args = args.get("color")
        size_from_args = args.get("size")

        if pending_action == "need_color":
            color_input = args.get("color")
            product = pending_data.get("product")
            quantity = pending_data.get("quantity", 1)
            
            # Normalize the color if provided
            if color_input:
                valid_colors = database.get_colors(client_id, product)
                if valid_colors:
                    color = normalize_product_name(color_input, valid_colors)
                    if not color:
                        # Color provided but not valid
                        state_manager.clear_pending_action(client_id, sender_id)
                        return {
                            "status": "fail",
                            "message": f"Sorry, {color_input} isn't available for {product}",
                            "available_colors": valid_colors
                        }
            
            state_manager.clear_pending_action(client_id, sender_id)

        elif pending_action == "need_size":
            size_input = args.get("size")
            product = pending_data.get("product")
            color = pending_data.get("color")
            quantity = pending_data.get("quantity", 1)
            
            # Normalize the size if provided
            if size_input:
                valid_sizes = database.get_sizes(client_id, product)
                if valid_sizes:
                    size = normalize_product_name(size_input, valid_sizes)
                    if not size:
                        # Size provided but not valid
                        state_manager.clear_pending_action(client_id, sender_id)
                        return {
                            "status": "fail",
                            "message": f"Sorry, {size_input} isn't available for {product}",
                            "available_sizes": valid_sizes
                        }
                else:
                    size = None  # No sizes required for this product
            
            state_manager.clear_pending_action(client_id, sender_id)
        
        # Get product from args
        product_name = args.get("product_name")
        if not product and product_name:
            product = normalize_product_name(product_name, valid_products)
        
        if not product:
            return {
                "status": "fail",
                "message": f"Sorry, we don't have {product_name or 'that item'}",
                "suggestions": valid_products
            }
        
        response["product"] = product
    
        # Get quantity from args
        try:
            quantity = int(args.get("quantity", 1))
        except (TypeError, ValueError):
            quantity = 1
        
        response["quantity"] = quantity
        
        # If color wasn't set by pending_action handling, try to get it from args
        if not color and color_from_args:
            valid_colors = database.get_colors(client_id, product)
            color = normalize_product_name(color_from_args, valid_colors)
        
        # If still no color, ask for it
        if not color:
            valid_colors = database.get_colors(client_id, product)
            
            if not valid_colors:
                # Product doesn't have color options, skip
                color = None
            else:
                state_manager.set_pending_action(client_id, sender_id, "need_color", {
                    "product": product,
                    "quantity": quantity
                })
                return {
                    "status": "need_color",
                    "available_colors": valid_colors
                }
        
        # If size wasn't set by pending_action handling, try to get it from args
        if not size and size_from_args:
            valid_sizes = database.get_sizes(client_id, product)
            if valid_sizes:
                size = extract_size(size_from_args, valid_sizes)
        
        # If still no size, check if product requires size
        if not size:
            valid_sizes = database.get_sizes(client_id, product)
            
            if valid_sizes:
                # Product has size options, ask for it
                state_manager.set_pending_action(client_id, sender_id, "need_size", {
                    "product": product,
                    "color": color,
                    "quantity": quantity
                })
                return {
                    "status": "need_size",
                    "available_sizes": valid_sizes
                }
            else:
                # Product doesn't have size options
                size = None


        response["color"] = color
        response["size"] = size
        
        item = {"name": product, "quantity": quantity, "color": color, "size": size}

        
        state_manager.add_item_to_cart(client_id, sender_id, item)
        print(f"Updated inprogress_order for client {client_id}")
        response.update({
            "status": "success",
            "message": f"Added {quantity} {product} in {color} to your cart"
        })
    
        status = response["status"]
        if status == "success":
            cart = state_manager.get_active_cart(client_id, sender_id)
            response["cart"] = cart
            response["next_step"] = "confirm_add"  

            return response

        elif status == "need_color":
            response["next_step"] = "request_color"
            response["available_colors"] = valid_colors
            return response
        
        elif status == "need_size":
            response["next_step"] = "request_size"
            response["available_sizes"] = valid_sizes
            return response
        
        elif status == "fail":
            return response
    
    elif name == "order_confirm":
        confirm = args.get("confirmation")
        if confirm:
            state_manager.set_pending_action(client_id, sender_id, "need_customer_info", {
                "collected": {}
            })
            return {
                "status": "need_info",
                "message": "Please provide your full name:",
                "next_field": "name"
            }
        else:
            return {
                "status": "cancelled",
                "message": "Order not confirmed. Your cart has been saved."
            }
        
    elif name == "collect_customer_info":
        response = {
            "status": "incomplete",
            "message": "",
            "next_field": None
        }
        
        context = state_manager.get_conversation_context(client_id, sender_id)
        if context.get("pending_action") != "need_customer_info":
            return {
                "status": "error",
                "message": "No active customer info collection session."
            }
        
        collected = context.get("pending_data", {}).get("collected", {})
        
        if args.get("name") and not collected.get("name"):
            collected["name"] = args["name"]
        if args.get("phone") and not collected.get("phone"):
            collected["phone"] = args["phone"]
        if args.get("address") and not collected.get("address"):
            collected["address"] = args["address"]
        
        missing_fields = []
        if not collected.get("name"):
            missing_fields.append("name")
        if not collected.get("phone"):
            missing_fields.append("phone")
        if not collected.get("address"):
            missing_fields.append("address")
        
        if missing_fields:
            next_field = missing_fields[0]
            prompts = {
                "name": "Please confirm your full name:",
                "phone": "Please share your phone number:",
                "address": "Where should we deliver your order?"
            }
            
            # Update collected info in pending data
            context["pending_data"]["collected"] = collected
            context["pending_data"]["next_field"] = next_field
            state_manager.update_conversation_context(client_id, sender_id, {"pending_data": context["pending_data"]})
            
            current_value = collected.get(next_field, "")
            prompt = prompts[next_field]
            if current_value:
                prompt = f"{prompt} (Current: {current_value})"
            
            return {
                "status": "need_info",
                "message": prompt,
                "next_field": next_field
            }
        
        customer_info = {
            "name": collected["name"],
            "phone": collected["phone"],
            "address": collected["address"]
        }
        
        items = state_manager.get_active_cart(client_id, sender_id)
        order_id = save_to_db(client_id, items, customer_info)
        total_price = database.get_total_price(client_id, order_id)
        
        state_manager.clear_pending_action(client_id, sender_id)
        state_manager.clear_context(client_id, sender_id)
        
        # Clear conversation history to start fresh
        # (Handled by clear_context above, but ensuring double check if needed or just rely on clear_context)
        # state_manager.clear_context already clears history.
        
        return {
            "status": "success",
            "order_id": str(order_id),
            "total_price": str(total_price),
            "message": f"Order confirmed! 🎉\nOrder ID: {order_id}\nTotal: Rs.{total_price:.2f}"
        }
    
    elif name == "track_order":
        context = state_manager.get_conversation_context(client_id, sender_id)
        pending_action = context.get("pending_action")
        
        if pending_action == "need_order_id":
            order_id = args.get("order_id")
            state_manager.clear_pending_action(client_id, sender_id)
        else:
            order_id = args.get("order_id")
        
        print(f"\n\nTHE ORDER ID IS: '{order_id}' (Type: {type(order_id)})\n\n")
        
        # Only proceed if we have a valid, non-empty order ID
        if not order_id:
            state_manager.set_pending_action(client_id, sender_id, "need_order_id", {
                "intent": "track_order"
            })
            return {
                "status": "need_info",
                "message": "Sure! Please provide your order ID so I can check its status.",
            }
        
        status = database.get_status(client_id, order_id)
    
        if not status:
            return {
                "status": "fail",
                "message": f"Order #{order_id} not found. Please double-check your order ID."
            }
        
        return {
            "status": "success",
            "message": f"Your order #{order_id} is {status}.",
        }
    
    elif name == "remove_item":
        context = state_manager.get_conversation_context(client_id, sender_id)
        cart_items = state_manager.get_active_cart(client_id, sender_id)
        
        if not cart_items:
            return {
                "status": "fail",
                "message": "Your cart is empty. Nothing to remove!",
            }
        
        # cart_items is already retrieved above
        item_name = normalize_product_name(args.get("product_name", ""), valid_products)

        matches = []
        for idx, item in enumerate(cart_items):
            if item_name == item["name"]:
                matches.append((idx, item))
        
        if not matches:
            item_list = "\n".join([f"{i+1}. {item['name']} ({item['color']}) x {item['quantity']}" 
                                for i, item in enumerate(cart_items)])
            return {
                "status": "need_info",
                "message": f"No '{item_name}' in your cart. Current items:\n{item_list}"
            }
        
        if len(matches) == 1:
            idx, item = matches[0]
            # We need to remove by index from the actual cart list
            # Since cart_items is a reference to the list in context, popping from it should work if it's the same object
            # But better to use the manager method to be safe if we implement it, or just pop here since we have the reference.
            # However, matches[0] idx is index in cart_items.
            
            removed_item = state_manager.remove_item_from_cart(client_id, sender_id, idx)
            
            # Check if cart is empty is handled by get_active_cart returning empty list next time
            
            return {
                "status": "success",
                "message": f"Removed {removed_item['quantity']} {removed_item['name']} ({removed_item['color']}) from your cart",
                "cart": cart_items
            }
        
        if len(matches) > 1:
            state_manager.set_pending_action(client_id, sender_id, "need_remove_specification", {
                "matches": matches,
                "item_name": item_name
            })
            
            match_list = "\n".join([f"{i+1}. {item['name']} ({item['color']}) x {item['quantity']}" 
                                for i, (idx, item) in enumerate(matches)])
            return {
                "status": "need_info",
                "message": f"Multiple '{item_name}' items found:\n{match_list}\n\nWhich one would you like to remove? (Reply with number)",
                "options": [str(i+1) for i in range(len(matches))]
            }
        
    elif name == "order_cancel":
        context = state_manager.get_conversation_context(client_id, sender_id)
        pending_action = context.get("pending_action")
        
        if pending_action == "need_order_id":
            order_id = args.get("order_id")
            state_manager.clear_pending_action(client_id, sender_id)
        else: 
            order_id = args.get("order_id")

        if not order_id:
            state_manager.set_pending_action(client_id, sender_id, "need_order_id", {
                "intent": "order_cancel"
            })
            return {
                "status": "need_info",
                "message": "Sure! Please provide your order ID so I can check its status.",
            }
        
        status = database.get_status(client_id, order_id)
        print(status)
        if not status:
            return {
                "status": "fail",
                "message": f"Order #{order_id} not found. Please double-check your order ID."
            }
        
        if status.lower() == "pending":
            rcode = database.cancel_order(client_id, order_id)
            if rcode == 1:
                # Clear conversation history to start fresh
                state_manager.clear_context(client_id, sender_id)
                
                return {
                    "status": "success",
                    "message": f"Order #{order_id} has been canceled"
                }
        else:
            return {
                "status": "Unable to cancel",
                "message": f"Its already {status}"
            }

