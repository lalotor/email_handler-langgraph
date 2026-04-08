"""Unit tests for agents/state.py module.

Tests the EmailAgentState and EmailClassification TypedDict structures
to ensure proper type definitions and data validation.
"""

import pytest
from typing import get_type_hints
from agents.state import EmailAgentState, EmailClassification


class TestEmailClassification:
    """Test suite for EmailClassification TypedDict."""

    def test_email_classification_structure(self):
        """Test that EmailClassification has the correct structure with all required fields.
        
        Validates that the TypedDict contains intent, urgency, topic, and summary fields.
        """
        # Create a valid classification
        classification: EmailClassification = {
            "intent": "question",
            "urgency": "medium",
            "topic": "password reset",
            "summary": "User needs help resetting password"
        }
        
        assert classification["intent"] == "question"
        assert classification["urgency"] == "medium"
        assert classification["topic"] == "password reset"
        assert classification["summary"] == "User needs help resetting password"

    def test_email_classification_all_intent_values(self):
        """Test that all valid intent literal values are accepted.
        
        Ensures the intent field accepts: question, bug, billing, feature, complex.
        """
        valid_intents = ["question", "bug", "billing", "feature", "complex"]
        
        for intent in valid_intents:
            classification: EmailClassification = {
                "intent": intent,
                "urgency": "medium",
                "topic": "test",
                "summary": "test summary"
            }
            assert classification["intent"] == intent

    def test_email_classification_all_urgency_values(self):
        """Test that all valid urgency literal values are accepted.
        
        Ensures the urgency field accepts: low, medium, high, critical.
        """
        valid_urgencies = ["low", "medium", "high", "critical"]
        
        for urgency in valid_urgencies:
            classification: EmailClassification = {
                "intent": "question",
                "urgency": urgency,
                "topic": "test",
                "summary": "test summary"
            }
            assert classification["urgency"] == urgency

    def test_email_classification_topic_string(self):
        """Test that topic field accepts any string value.
        
        Validates that topic is a flexible string field without restrictions.
        """
        topics = [
            "password reset",
            "billing issue",
            "feature request: dark mode",
            "bug: login failure",
            "",  # Empty string should be valid
            "a" * 1000  # Very long topic
        ]
        
        for topic in topics:
            classification: EmailClassification = {
                "intent": "question",
                "urgency": "medium",
                "topic": topic,
                "summary": "test"
            }
            assert classification["topic"] == topic

    def test_email_classification_summary_string(self):
        """Test that summary field accepts any string value.
        
        Validates that summary is a flexible string field for detailed descriptions.
        """
        summaries = [
            "Short summary",
            "A much longer summary with multiple sentences. This describes the email in detail.",
            "",
            "Summary with special chars: @#$%^&*()"
        ]
        
        for summary in summaries:
            classification: EmailClassification = {
                "intent": "question",
                "urgency": "medium",
                "topic": "test",
                "summary": summary
            }
            assert classification["summary"] == summary


class TestEmailAgentState:
    """Test suite for EmailAgentState TypedDict."""

    def test_email_agent_state_minimal_structure(self, sample_email_state):
        """Test that EmailAgentState can be created with minimal required fields.
        
        Validates that a state can be created with email_content, sender_email, 
        email_id, and optional fields set to None.
        """
        state = sample_email_state
        
        assert state["email_content"] == "I need help resetting my password"
        assert state["sender_email"] == "user@example.com"
        assert state["email_id"] == "test_email_123"
        assert state["classification"] is None
        assert state["search_results"] is None
        assert state["customer_history"] is None
        assert state["draft_response"] is None

    def test_email_agent_state_with_classification(self, sample_email_state, sample_classification):
        """Test that EmailAgentState correctly stores classification data.
        
        Validates that the classification field can hold an EmailClassification dict.
        """
        state = sample_email_state
        state["classification"] = sample_classification
        
        assert state["classification"] is not None
        assert state["classification"]["intent"] == "question"
        assert state["classification"]["urgency"] == "medium"
        assert state["classification"]["topic"] == "password reset"

    def test_email_agent_state_with_search_results(self, sample_email_state):
        """Test that EmailAgentState correctly stores search results.
        
        Validates that search_results can hold a list of document strings.
        """
        state = sample_email_state
        state["search_results"] = [
            "Reset password via Settings > Security",
            "Password must be 12+ characters",
            "Include uppercase, lowercase, numbers, symbols"
        ]
        
        assert len(state["search_results"]) == 3
        assert "Settings > Security" in state["search_results"][0]

    def test_email_agent_state_with_customer_history(self, sample_email_state):
        """Test that EmailAgentState correctly stores customer history data.
        
        Validates that customer_history can hold arbitrary customer data as a dict.
        """
        state = sample_email_state
        state["customer_history"] = {
            "tier": "premium",
            "account_age_days": 365,
            "previous_tickets": 2,
            "satisfaction_score": 4.5
        }
        
        assert state["customer_history"]["tier"] == "premium"
        assert state["customer_history"]["account_age_days"] == 365
        assert state["customer_history"]["satisfaction_score"] == 4.5

    def test_email_agent_state_with_draft_response(self, sample_email_state):
        """Test that EmailAgentState correctly stores draft response.
        
        Validates that draft_response can hold the generated email response text.
        """
        state = sample_email_state
        state["draft_response"] = "Thank you for contacting us. To reset your password..."
        
        assert state["draft_response"] is not None
        assert "reset your password" in state["draft_response"]

    def test_email_agent_state_empty_search_results(self, sample_email_state):
        """Test that EmailAgentState handles empty search results list.
        
        Validates that search_results can be an empty list when no docs are found.
        """
        state = sample_email_state
        state["search_results"] = []
        
        assert state["search_results"] == []
        assert len(state["search_results"]) == 0

    def test_email_agent_state_empty_customer_history(self, sample_email_state):
        """Test that EmailAgentState handles empty customer history dict.
        
        Validates that customer_history can be an empty dict for new customers.
        """
        state = sample_email_state
        state["customer_history"] = {}
        
        assert state["customer_history"] == {}
        assert len(state["customer_history"]) == 0

    def test_email_agent_state_messages_list(self, sample_email_state):
        """Test that EmailAgentState correctly stores messages list.
        
        Validates that messages field can hold a list of message strings.
        """
        state = sample_email_state
        state["messages"] = [
            "Processing email: I need help",
            "Classification complete",
            "Draft generated"
        ]
        
        assert len(state["messages"]) == 3
        assert state["messages"][0].startswith("Processing email")

    def test_email_agent_state_complete_workflow(self):
        """Test EmailAgentState through a complete workflow simulation.
        
        Validates that state can be progressively updated through all workflow stages.
        """
        # Initial state
        state: EmailAgentState = {
            "email_content": "I can't log in to my account",
            "sender_email": "user@test.com",
            "email_id": "email_999",
            "classification": None,
            "search_results": None,
            "customer_history": None,
            "draft_response": None,
            "messages": []
        }
        
        # After classification
        state["classification"] = {
            "intent": "bug",
            "urgency": "high",
            "topic": "login failure",
            "summary": "User unable to access account"
        }
        
        # After search
        state["search_results"] = [
            "Check browser cache and cookies",
            "Try password reset"
        ]
        
        # After draft
        state["draft_response"] = "We're sorry you're having login issues..."
        
        # Validate complete state
        assert state["email_id"] == "email_999"
        assert state["classification"]["intent"] == "bug"
        assert len(state["search_results"]) == 2
        assert state["draft_response"] is not None

    def test_email_agent_state_special_characters_in_content(self):
        """Test that EmailAgentState handles special characters in email content.
        
        Validates that email_content can contain various special characters and unicode.
        """
        state: EmailAgentState = {
            "email_content": "Hello! I have a question about pricing: $99/month? That's 50% more than before! 😊",
            "sender_email": "test@example.com",
            "email_id": "special_123",
            "classification": None,
            "search_results": None,
            "customer_history": None,
            "draft_response": None,
            "messages": []
        }
        
        assert "$99/month" in state["email_content"]
        assert "😊" in state["email_content"]
        assert "50%" in state["email_content"]

    def test_email_agent_state_long_email_content(self):
        """Test that EmailAgentState handles very long email content.
        
        Validates that email_content can store lengthy customer emails.
        """
        long_content = "This is a very long email. " * 500  # ~14,000 characters
        
        state: EmailAgentState = {
            "email_content": long_content,
            "sender_email": "verbose@example.com",
            "email_id": "long_email_001",
            "classification": None,
            "search_results": None,
            "customer_history": None,
            "draft_response": None,
            "messages": []
        }
        
        assert len(state["email_content"]) > 10000
        assert state["email_content"].startswith("This is a very long email.")

    def test_email_agent_state_multiple_email_addresses(self):
        """Test that EmailAgentState handles various email address formats.
        
        Validates that sender_email accepts different valid email formats.
        """
        email_addresses = [
            "simple@example.com",
            "user.name+tag@example.co.uk",
            "test_user123@subdomain.example.com",
            "a@b.c"  # Minimal valid email
        ]
        
        for email in email_addresses:
            state: EmailAgentState = {
                "email_content": "test",
                "sender_email": email,
                "email_id": "test",
                "classification": None,
                "search_results": None,
                "customer_history": None,
                "draft_response": None,
                "messages": []
            }
            assert state["sender_email"] == email
