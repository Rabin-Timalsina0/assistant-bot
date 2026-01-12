from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from app import helper_funcs, functions, prompts, database, state_manager
from app.conversation_handler import ConversationStateHandler
from app.admin_api import router as admin_router
from collections import deque
import json
import os
from contextlib import asynccontextmanager
import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def get_client_config(client_id: int):
    try:
        client_info = database.get_clients(client_id)
        if not client_info:
            raise HTTPException(status_code=404, detail="Client not found")
        return {
            "verify_token": client_info[2],
            "api_key": client_info[3],
            "page_access_token": client_info[4],
            "openai_model": client_info[5]
        }
    except Exception as e:
        print(f"Database error in get_client_config: {e}")
        try:
            client_info = database.get_clients(client_id)
            if not client_info:
                raise HTTPException(status_code=404, detail="Client not found")
            return {
                "verify_token": client_info[2],
                "api_key": client_info[3],
                "page_access_token": client_info[4],
                "openai_model": client_info[5]
            }
        except Exception as retry_error:
            print(f"Database retry failed in get_client_config: {retry_error}")
            raise HTTPException(status_code=503, detail="Database connection error")


def get_openai_client(api_key: str):
    return OpenAI(api_key=api_key)

def intent_classifier(client: OpenAI, user_input: str, previous_bot_message: str = None):
    messages = [
        {"role": "system", "content": prompts.intent_prompt}
    ]
    if previous_bot_message:
        messages.append({"role": "assistant", "content": previous_bot_message})
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fixed: valid OpenAI model
        messages=messages,
    )
    return response.choices[0].message.content

default_intents = {
    "greeting": helper_funcs.greeting(),
    "inventory_inquiry": helper_funcs.inventory_inquiry,
    "start_order": "Sure, what would you like to order? Here are our available products: ",
    "handle_faqs": helper_funcs,
    "conversation_continuation": "",
    "default": ""
}

@app.get("/webhook/{client_id}")
async def verify_webhook(client_id: str, request: Request):
    config = get_client_config(client_id)
    mode = request.query_params.get("hub.mode")
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and verify_token == config["verify_token"]:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")
    
@app.post("/webhook/{client_id}")
async def get_message(client_id: str, request: Request):
    config = get_client_config(client_id)
    client_openai = get_openai_client(config["api_key"])
    
    default_intents["start_order"] = (
        "Sure, what would you like to order? Here are our available products: " + 
        ", ".join(database.get_names(client_id))
    )
    sender_id = None
    current_state = {}
    data = await request.json()
    print("\n\n\nReceived webhook data:", data, "\n\n\n")
    for entry in data.get("entry", []):
        for messaging in entry.get("messaging", []):
            # Skip read receipts and other non-message events
            if "message" not in messaging:
                if "read" in messaging:
                    print("Read receipt received, ignoring")
                continue  # skip this event

            message_data = messaging.get("message", {})
            sender_id = messaging["sender"]["id"]

            if message_data.get("is_echo"):
                print("Echo message received, ignoring")
                continue   
            if sender_id == str(INSTAAGRAM_PAGE_ID):
                continue
            
            message_id = messaging.get("message", {}).get("mid")
            if not message_id:
                continue  # No message ID, skip
            if message_id in state_manager.processed_messages[client_id]:
                print(f"Duplicate message detected: {message_id}")
                continue
            state_manager.processed_messages[client_id].add(message_id)

            user_text = None
            
            # Handle text
            if "message" in messaging and "text" in messaging["message"]:
                user_text = messaging["message"]["text"]

            # Handle image (image search disabled - like chatbot_NoImage)
            if "message" in messaging and "attachments" in messaging["message"]:
                attachments = messaging["message"]["attachments"]
                for attachment in attachments:
                    if attachment.get("type") == "image":
                        image_url = attachment.get("payload", {}).get("url")
                        print(f"Received image: {image_url} (image search disabled)")
                        response = "Image search is currently unavailable. Please describe what you're looking for in text."
                        helper_funcs.send_message_facebook(sender_id, response, access_token=config["page_access_token"])
                        return {"status": "success"}
                    
                    elif attachment.get("type") == "share" or attachment.get("type") == "ig_reel":
                        caption = attachment.get("payload", {}).get("title", "")
                        if caption:
                            product = helper_funcs.extract_from_caption(caption, client_id)
                            if product:
                                response = database.get_desc(client_id, product)
                                helper_funcs.send_message_facebook(sender_id, response, access_token=config["page_access_token"])
                                
                                # Store product in conversation history for context
                                if sender_id not in state_manager.conversation_histories[client_id]:
                                    state_manager.conversation_histories[client_id][sender_id] = deque(maxlen=5)
                                state_manager.conversation_histories[client_id][sender_id].append({
                                    "role": "user", 
                                    "content": f"Shared post about {product}"
                                })
                                state_manager.conversation_histories[client_id][sender_id].append({
                                    "role": "assistant", 
                                    "content": response
                                })
                                
                                return {"status": "success"}
                            else:
                                response = "How can I help with the post?"
                                helper_funcs.send_message_facebook(sender_id, response, access_token=config["page_access_token"])
                                return {"status": "success"}
                        else:
                            response = "How can I help with the post?"
                            helper_funcs.send_message_facebook(sender_id, response, access_token=config["page_access_token"])
                            return {"status": "success"}

    try:
        # Skip processing if no text message (e.g., read receipts, echo messages)
        if user_text is None:
            return {"status": "success"}
        
        # 1. Extract and Update Context (Persistent Product Memory)
        helper_funcs.extract_and_update_context(client_id, sender_id, user_text)

        # 2. Check for pending actions (like waiting for color/size/info)
        context = state_manager.get_conversation_context(client_id, sender_id)
        pending_action = context.get("pending_action")
        pending_data = context.get("pending_data", {})

        skip_intent = False
        intent = None
        args = {}

        # ... (rest of pending action logic) ...

        # Natural Conversation Logic:
        # Check if the user input satisfies the pending action.
        # If yes, set intent and skip_intent = True.
        # If no (e.g. it's a question), let it fall through to intent_classifier.

        if pending_action == "need_command":
            # For need_command, we assume any input is the command unless it looks like something else?
            # Actually, need_command usually comes from "start_order" or similar where we expect a product name.
            # Let's try to normalize it as a product.
            # If it's a product, good. If not, maybe it's a question.
            # But wait, need_command logic in original code was:
            # intent = "item_inquiry", args = {"product_name": current_state["product_name"]}
            # It seems it was auto-triggering item_inquiry with a stored product name?
            # Ah, looking at original code:
            # if current_state.get("state") == "need_command":
            #     intent = "item_inquiry"
            #     args = {"product_name": current_state["product_name"]}
            #     skip_intent = True
            #     del helper_funcs.user_states...
            # This looks like it was handling a state where we already knew the product but needed to trigger inquiry?
            # Let's keep it as is for now but use new state manager.
            
            intent = "item_inquiry"
            args = {"product_name": pending_data.get("product_name")}
            skip_intent = True
            state_manager.clear_pending_action(client_id, sender_id)

        elif pending_action == "need_color":
            valid_colors = database.get_colors(client_id, pending_data.get("product"))
            color = helper_funcs.extract_color(user_text, valid_colors)
            
            if color:
                intent = "add_item"
                args = {
                    "product_name": pending_data.get("product"),
                    "quantity": pending_data.get("quantity"),
                    "color": color
                }
                skip_intent = True
            else:
                # User said something that isn't a color. Let AI handle it.
                # The pending_action remains "need_color" in the context.
                print(f"User input '{user_text}' is not a color. Letting AI handle it.")
        
        elif pending_action == "need_size":
            valid_sizes = database.get_sizes(client_id, pending_data.get("product"))
            size = helper_funcs.extract_size(user_text, valid_sizes)
            
            if size:
                intent = "add_item"
                args = {
                    "product_name": pending_data.get("product"),
                    "quantity": pending_data.get("quantity"),
                    "color": pending_data.get("color"),
                    "size": size
                }
                skip_intent = True
            else:
                print(f"User input '{user_text}' is not a size. Letting AI handle it.")
        
        elif pending_action == "need_customer_info":
            next_field = pending_data.get("next_field", "name")
            extracted_value = helper_funcs.extract_customer_info(user_text, next_field)
            
            # For customer info, it's harder to validate strictness. 
            # But if they ask a question, extract_customer_info might just return the whole text or nothing.
            # Let's assume if it looks like a question, we might want to answer it.
            # But for now, let's keep it simple: if extract_customer_info returns something different than input or we are confident?
            # Actually, the original logic was: extracted_value = ...; intent = "collect_customer_info"
            # We should probably allow questions here too.
            # If the user asks "Why do you need my phone?", we want to answer.
            # But extract_customer_info is regex based.
            
            is_question = "?" in user_text or any(w in user_text.lower() for w in ["why", "what", "how", "where"])
            if is_question and len(user_text.split()) > 2: # Simple heuristic
                 print(f"User input '{user_text}' looks like a question. Letting AI handle it.")
            else:
                intent = "collect_customer_info"
                args = {
                    next_field: extracted_value
                }
                skip_intent = True

        elif pending_action == "need_order_id":
            order_id = helper_funcs.extract_order_id(user_text)
            if order_id:
                intent = pending_data.get("intent")
                args = {"order_id": order_id}
                skip_intent = True
            else:
                 print(f"User input '{user_text}' is not an order ID. Letting AI handle it.")
        
        elif pending_action == "potential_order":
            affirmative = ["yes", "yeah", "sure", "ok", "okay", "i want it", "add it", "buy it"]
            if any(word in user_text.lower() for word in affirmative):
                intent = "add_item"
                args = {
                    "product_name": pending_data.get("product"),
                    "quantity": pending_data.get("quantity", 1),
                    "color": pending_data.get("color"),
                    "size": pending_data.get("size")
                }
                skip_intent = True
                state_manager.clear_pending_action(client_id, sender_id)
            else:
                # If they say no or something else, clear the potential order
                state_manager.clear_pending_action(client_id, sender_id)

        if sender_id not in state_manager.conversation_histories[client_id]:
            state_manager.conversation_histories[client_id][sender_id] = deque(maxlen=5)
            print(f"New user: {sender_id} for client: {client_id}")
            recent_bot_msg = None
        else:
            if state_manager.conversation_histories[client_id][sender_id]:
                for msg in reversed(state_manager.conversation_histories[client_id][sender_id]):
                    if msg["role"] == "assistant":
                        recent_bot_msg = msg["content"]
                        break
                else:
                    recent_bot_msg = None
            else:
                recent_bot_msg = None
        
        print(f"Current history for client {client_id}:", list(state_manager.conversation_histories[client_id].get(sender_id, [])))
        
        # Check if user has an active cart and wants to checkout
        active_cart = state_manager.get_active_cart(client_id, sender_id)
        has_active_cart = bool(active_cart)
        checkout_phrases = ["that's it", "thats it", "that is it", "proceed", "checkout", 
                          "yes proceed", "confirm", "place order", "i'm done", "im done", 
                          "that's all", "thats all", "ready", "go ahead", "2", "option 2"]
        user_lower = user_text.lower().strip() if user_text else ""
        
        if not skip_intent:
            # If user has active cart and says checkout phrase, skip intent classification
            if has_active_cart and any(phrase in user_lower for phrase in checkout_phrases):
                intent = "order_confirm"
                skip_intent = True
                args = {"confirmation": True}
                print(f"Detected checkout intent for client {client_id} due to active cart and checkout phrase")
            else:
                intent = intent_classifier(client_openai, user_text, recent_bot_msg)
                # Override conversation_continuation if there's a cart and user wants to proceed
                if intent == "conversation_continuation" and has_active_cart:
                    if any(phrase in user_lower for phrase in checkout_phrases):
                        intent = "order_confirm"
                        skip_intent = True
                        args = {"confirmation": True}
                        print(f"Overriding intent to order_confirm for client {client_id} due to active cart and checkout phrase")
        
        print(f"Detected intent for client {client_id}: {intent}")
        
        state_manager.conversation_histories[client_id][sender_id].append({"role": "user", "content": user_text})
        
        if intent in default_intents:
            if intent == "inventory_inquiry":
                response = default_intents[intent](client_id)
            else:
                response = default_intents[intent]
            
            if intent == "start_order":
                state_manager.conversation_histories[client_id][sender_id].append({"role": "assistant", "content": response})

            elif intent == "handle_faqs":
                func = default_intents[intent]
                response = func.handle_faqs(user_text)
                state_manager.conversation_histories[client_id][sender_id].append({"role": "assistant", "content": response})
            
            elif intent == "default":
                messages = [
                    {"role": "system", "content": prompts.system_prompt},
                    *list(state_manager.conversation_histories[client_id][sender_id])
                ]
                gpt_message = client_openai.chat.completions.create(
                    model=config["openai_model"],
                    messages=messages
                )
                response = gpt_message.choices[0].message.content
                state_manager.conversation_histories[client_id][sender_id].append({"role": "assistant", "content": response})

            else:
                state_manager.conversation_histories[client_id][sender_id].append({"role": "assistant", "content": response})

            # Send the main response first
            helper_funcs.send_message_facebook(sender_id, response, access_token=config["page_access_token"])

            # Use ConversationStateHandler to handle interruptions and resumptions
            handler = ConversationStateHandler(client_id, sender_id, client_openai, config)
            should_resume, resumption_msg = handler.process_message(user_text, intent, response)
            
            if should_resume and resumption_msg:
                # Send resumption message as a separate message
                helper_funcs.send_message_facebook(sender_id, resumption_msg, access_token=config["page_access_token"])
                
                # Add to conversation history
                state_manager.conversation_histories[client_id][sender_id].append({
                    "role": "assistant", 
                    "content": resumption_msg
                })

            return {"status": "success"}
        
        else:
            print("Inprogess orders: ", state_manager.get_active_cart(client_id, sender_id))
            messages = [
                {"role": "system", "content": prompts.system_prompt},
                *list(state_manager.conversation_histories[client_id][sender_id])
            ]
            
            print("Messages to GPT:", messages)
            
            json_func = []
            for func in functions.ALL_FUNCTIONS:
                if func["name"] == intent:
                    json_func = [func]
                    break
            
            if not skip_intent:
                response = client_openai.chat.completions.create(
                    model=config["openai_model"],
                    messages=messages,
                    functions=json_func,
                    function_call={"name": intent}
                )
                gpt_message = response.choices[0].message
            elif skip_intent:
                class SimulatedFunctionCall:
                    def __init__(self, name, arguments):
                        self.name = name
                        self.arguments = json.dumps(arguments)
                
                class SimulatedMessage:
                    def __init__(self, function_call):
                        self.function_call = function_call
                        self.content = None
                
                gpt_message = SimulatedMessage(SimulatedFunctionCall(intent, args))

            if hasattr(gpt_message, 'function_call') and gpt_message.function_call:
                print("Function call detected")
                func_name = gpt_message.function_call.name
                if skip_intent:
                    func_arg = args
                else:
                    func_arg = json.loads(gpt_message.function_call.arguments)

                if func_name == "pic_samples":
                    product_name = func_arg.get("product_name")
                    product = helper_funcs.normalize_product_name(
                        product_name, 
                        database.get_names(client_id)
                    )

                    color_name = func_arg.get("color")
                    color = None
                    if color_name:
                        color = helper_funcs.normalize_product_name(
                            color_name,
                            database.get_colors(client_id, product)
                        )

                    urls = database.get_urls(client_id, product, color)
                    colors = database.get_colors(client_id, product)
                    response = f"Of course. We have {product} in {', '.join(colors)}"
                    if urls:
                        helper_funcs.send_message_facebook(sender_id, response, urls, access_token=config["page_access_token"])
                    else:
                        response = f"Sorry, currently we dont have images of {product}"
                        helper_funcs.send_message_facebook(sender_id, response, access_token=config["page_access_token"])
                    return {"status": "success"}

                # For add_item, merge with persistent context if args are missing
                if func_name == "add_item":
                    current_context_item = state_manager.get_product_context(client_id, sender_id)
                    if not func_arg.get("product_name") and current_context_item.get("product"):
                        func_arg["product_name"] = current_context_item["product"]
                    if (not func_arg.get("color") or func_arg.get("color") == "any") and current_context_item.get("color"):
                        func_arg["color"] = current_context_item["color"]
                    if (not func_arg.get("size") or func_arg.get("size") == "any") and current_context_item.get("size"):
                        func_arg["size"] = current_context_item["size"]

                # Call the function handler once with the merged args
                func_response = helper_funcs.handle_function_call(
                    client_id, sender_id, func_name, func_arg
                )
                
                # Handle add_item response formatting BEFORE the second OpenAI call
                if func_name == "add_item":
                    if func_response.get("next_step") == "confirm_add":
                        cart = func_response.get("cart", [])
                        product = func_response["product"]
                        color = func_response["color"]
                        size = func_response["size"]
                        quantity = func_response["quantity"]
                        
                        cart_summary = "\n".join(
                            [f"- {item['quantity']} {item['name']} ({item['color']}) {item['size']}" 
                            for item in cart]
                        )
                        
                        response_content = (
                            f"Added {quantity} {product} in {color} of size {size} to your order!\n\n"
                            f"Current order:\n{cart_summary}\n\n"
                            "Would you like to:\n"
                            "1. Add another item\n"
                            "2. Proceed to checkout\n"
                            "3. Remove an item"
                        )
                        history_marker = f"added_item: {quantity} {product} ({color}, {size})"
                        
                        # Clear product context so next "add item" starts fresh
                        state_manager.clear_product_context(client_id, sender_id)
                        
                        # Send the response immediately and return
                        if history_marker:
                            state_manager.conversation_histories[client_id][sender_id].append({
                                "role": "assistant", 
                                "content": history_marker
                            })
                        helper_funcs.send_message_facebook(sender_id, response_content, access_token=config["page_access_token"])
                        return {"status": "success"}

                        
                    elif func_response.get("next_step") == "request_color":
                        response_content = (
                            f"Great choice! What color would you like for the {func_response['product']}?\n"
                            f"Available: {', '.join(database.get_colors(client_id, product))}"
                        )
                        history_marker = f"requested_color: {func_response['product']}"
                        
                        # Send the response immediately and return
                        if history_marker:
                            helper_funcs.conversation_histories[client_id][sender_id].append({
                                "role": "assistant", 
                                "content": history_marker
                            })
                        helper_funcs.send_message_facebook(sender_id, response_content, access_token=config["page_access_token"])
                        return {"status": "success"}
                    
                    elif func_response.get("next_step") == "request_size":
                        response_content = (
                            f"What size would you like for the {func_response['product']}?\n"
                            f"Available: {', '.join(database.get_sizes(client_id, product))}"
                        )
                        history_marker = f"requested_size: {func_response['product']}"
                        
                        # Send the response immediately and return
                        if history_marker:
                            helper_funcs.conversation_histories[client_id][sender_id].append({
                                "role": "assistant", 
                                "content": history_marker
                            })
                        helper_funcs.send_message_facebook(sender_id, response_content, access_token=config["page_access_token"])
                        return {"status": "success"}
                
                # Only make the second OpenAI call if we didn't handle add_item specially
                second_response_messages = [
                    {"role": "system", "content": prompts.system_prompt},
                    {"role": "user", "content": user_text},
                    {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": func_name,
                            "arguments": json.dumps(func_arg)
                        }
                    },
                    {
                        "role": "function",
                        "name": func_name,
                        "content": json.dumps(func_response)
                    }
                ]
                
                second_response = client_openai.chat.completions.create(
                    model=config["openai_model"],
                    messages=second_response_messages
                )
                response_content = second_response.choices[0].message.content
                print("Generated response:", response_content)
                
                # Handle history marker for non-add_item functions
                history_marker = None
                
            else:
                response_content = gpt_message.content
                history_marker = None

            # For non-add_item function calls or regular responses, send the message
            if history_marker:
                state_manager.conversation_histories[client_id][sender_id].append({
                    "role": "assistant", 
                    "content": history_marker
                })
            helper_funcs.send_message_facebook(sender_id, response_content, access_token=config["page_access_token"])
            return {"status": "success"}

    except Exception as e:
        import traceback
        print(f"Error for client {client_id}: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        # Log the error but don't send generic error messages to users
        # Only send error messages for critical failures
        return {"status": "error", "details": str(e)}
