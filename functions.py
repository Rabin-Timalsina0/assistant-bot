item_inquiry = {
    "name": "item_inquiry",
    "description": "Provide information about specific product",
    "parameters": {
        "type": "object",
        "properties": {
            "product_name": {"type": "string"},
            "color": {"type": "string"},
            "size": {"type": "string"}
        },
        "required": ["product_name"]
    }
}

pic_samples = {
    "name": "pic_samples",
    "description": "Provide user with images",
    "parameters": {
        "type": "object",
        "properties": {
            "product_name": {"type": "string"},
            "color": {"type": "string"}
        },
        "required": ["product_name"]
    }
}

add_item = {
    "name": "add_item",
    "description": "Place an order in the online store",
    "parameters": {
        "type": "object",
        "properties": {
            "product_name": { "type": "string" },
            "quantity": { "type": "integer" },
            "color": {"type": "string", "default": "any"},
            "size": {"type": "string", "default": "any"}
        },
        "required": ["product_name"]
    }
}

order_confirm = {
    "name": "order_confirm",
    "description": "Ask whether to confirm the order",
    "parameters": {
        "type": "object",
        "properties": {
            "confirmation": { "type": "boolean" }
        },
        "required": ["confirmation"]
    }
}

customer_info = {
    "name": "customer_info",
    "description": "Client info",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "phone": {"type": "string"},
            "address": {"type": "string"}
        },
        "required": ["name", "phone", "address"]
    }
}

cancel_order = {
    "name": "cancel_order",
    "description": "Cancel an existing order",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": { "type": "integer" },
            "user_id": { "type": "integer" }
        },
        "required": ["order_id", "user_id"]
    }
}

track_order = {
    "name": "track_order",
    "description": "Provide user status if user dont give id keep it none",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "integer"}
        },
        "required": []
    }
}

remove_item = {
    "name": "remove_item",
    "description": "Remove the item from order",
    "parameters": {
        "type": "object",
        "properties": {
            "product_name": {"type": "string"}
        },
        "required": ["product_name"]
    }
}

order_cancel = {
    "name": "order_cancel",
    "description": "Cancel the entire order",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "integer"}
        },
        "required": []
    }
}

ALL_FUNCTIONS = [item_inquiry, add_item, cancel_order, order_confirm, customer_info, track_order, remove_item, order_cancel, pic_samples]

    