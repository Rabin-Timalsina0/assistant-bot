"""
Conversation State Handler

This module provides a clean, class-based approach to managing conversation state,
with clear separation between normal flow and interruption handling.
"""

from app import state_manager, helper_funcs


class ConversationStateHandler:
    """
    Handles conversation state and flow, including interruptions and resumptions.
    
    This class encapsulates all logic related to:
    - Detecting interruptions (when user asks something while we're waiting for input)
    - Deciding when to send resumption messages
    - Generating contextual follow-up messages
    """
    
    # Intents that are part of the order flow and shouldn't trigger resumption
    ORDER_FLOW_INTENTS = {
        "add_item",
        "order_confirm", 
        "collect_customer_info",
        "remove_item",
        "order_cancel"
    }
    
    def __init__(self, client_id, sender_id, openai_client, config):
        """
        Initialize the conversation state handler.
        
        Args:
            client_id: The client ID
            sender_id: The user/sender ID
            openai_client: OpenAI client instance for GPT calls
            config: Configuration dict with access tokens, etc.
        """
        self.client_id = client_id
        self.sender_id = sender_id
        self.openai_client = openai_client
        self.config = config
        
    def process_message(self, user_text, intent, response):
        """
        Main entry point for processing a message and determining if resumption is needed.
        
        Args:
            user_text: The user's message text
            intent: The detected intent
            response: The bot's response to the user's message
            
        Returns:
            tuple: (should_send_resumption: bool, resumption_message: str or None)
        """
        # Check if there's a pending action
        context = state_manager.get_conversation_context(self.client_id, self.sender_id)
        pending_action = context.get("pending_action")
        
        if not pending_action:
            # Normal flow - no pending action
            return self.handle_normal_flow()
        
        # There's a pending action - check if this is an interruption
        if self.is_interruption(intent):
            # User interrupted with a different question
            return self.handle_interruption(user_text, intent, response)
        else:
            # User is responding to the pending action (normal flow)
            return self.handle_normal_flow()
    
    def handle_normal_flow(self):
        """
        Handle message when there's no interruption (normal conversation flow).
        
        Returns:
            tuple: (False, None) - no resumption needed
        """
        return False, None
    
    def handle_interruption(self, user_text, intent, response):
        """
        Handle message when user interrupts a pending action.
        
        Args:
            user_text: The interrupting message from the user
            intent: The intent of the interrupting message
            response: The bot's response to the interruption
            
        Returns:
            tuple: (True, resumption_message) if resumption should be sent
                   (False, None) if no resumption needed or error occurred
        """
        context = state_manager.get_conversation_context(self.client_id, self.sender_id)
        pending_action = context.get("pending_action")
        pending_data = context.get("pending_data", {})
        
        # Generate resumption message
        try:
            resumption_msg = helper_funcs.generate_resumption_message(
                self.openai_client,
                pending_action,
                pending_data,
                user_text,
                response
            )
            
            if resumption_msg:
                return True, resumption_msg
            else:
                return False, None
                
        except Exception as e:
            print(f"Error in handle_interruption: {e}")
            return False, None
    
    def is_interruption(self, intent):
        """
        Check if the current intent represents an interruption to the pending action.
        
        An interruption is when:
        - There's a pending action (checked by caller)
        - The current intent is NOT part of the order flow
        
        Args:
            intent: The detected intent
            
        Returns:
            bool: True if this is an interruption, False otherwise
        """
        return intent not in self.ORDER_FLOW_INTENTS
    
    def get_pending_context(self):
        """
        Get the current pending action and associated data.
        
        Returns:
            tuple: (pending_action, pending_data)
        """
        context = state_manager.get_conversation_context(self.client_id, self.sender_id)
        return context.get("pending_action"), context.get("pending_data", {})
    
    def clear_pending(self):
        """Clear the pending action for this user."""
        state_manager.clear_pending_action(self.client_id, self.sender_id)
