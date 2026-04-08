"""Shared pytest fixtures for email_handler-langgraph tests.

Provides common test data, mocks, and configuration for all test modules.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.state import EmailAgentState, EmailClassification


@pytest.fixture
def sample_email_state() -> EmailAgentState:
    """Provides a basic email state for testing.
    
    Returns:
        EmailAgentState with minimal required fields populated.
    """
    return {
        "email_content": "I need help resetting my password",
        "sender_email": "user@example.com",
        "email_id": "test_email_123",
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": []
    }


@pytest.fixture
def urgent_billing_state() -> EmailAgentState:
    """Provides an urgent billing email state for testing.
    
    Returns:
        EmailAgentState representing an urgent billing issue.
    """
    return {
        "email_content": "I was charged twice for my subscription! This is urgent!",
        "sender_email": "customer@example.com",
        "email_id": "billing_email_456",
        "classification": {
            "intent": "billing",
            "urgency": "critical",
            "topic": "double charge",
            "summary": "Customer charged twice for subscription"
        },
        "search_results": None,
        "customer_history": {"tier": "premium"},
        "draft_response": None,
        "messages": []
    }


@pytest.fixture
def bug_report_state() -> EmailAgentState:
    """Provides a bug report email state for testing.
    
    Returns:
        EmailAgentState representing a bug report.
    """
    return {
        "email_content": "The login button doesn't work on mobile devices",
        "sender_email": "tester@example.com",
        "email_id": "bug_email_789",
        "classification": {
            "intent": "bug",
            "urgency": "high",
            "topic": "login button mobile",
            "summary": "Login button not working on mobile"
        },
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": []
    }


@pytest.fixture
def question_state() -> EmailAgentState:
    """Provides a question email state for testing.
    
    Returns:
        EmailAgentState representing a customer question.
    """
    return {
        "email_content": "How do I export my data?",
        "sender_email": "curious@example.com",
        "email_id": "question_email_101",
        "classification": {
            "intent": "question",
            "urgency": "low",
            "topic": "data export",
            "summary": "Customer asking about data export"
        },
        "search_results": [
            "Go to Settings > Data > Export",
            "Choose CSV or JSON format",
            "Download will be ready in 5 minutes"
        ],
        "customer_history": None,
        "draft_response": None,
        "messages": []
    }


@pytest.fixture
def sample_classification() -> EmailClassification:
    """Provides a sample email classification.
    
    Returns:
        EmailClassification with typical values.
    """
    return {
        "intent": "question",
        "urgency": "medium",
        "topic": "password reset",
        "summary": "User needs help resetting password"
    }


@pytest.fixture
def mock_llm():
    """Provides a mocked ChatOpenAI instance.
    
    Returns:
        Mock object simulating ChatOpenAI behavior.
    """
    mock = MagicMock()
    mock.invoke.return_value = Mock(content="This is a generated response")
    return mock


@pytest.fixture
def mock_structured_llm():
    """Provides a mocked structured LLM for classification.
    
    Returns:
        Mock object that returns EmailClassification dict.
    """
    mock = MagicMock()
    mock.invoke.return_value = {
        "intent": "question",
        "urgency": "medium",
        "topic": "password reset",
        "summary": "User needs help resetting password"
    }
    return mock


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Sets up mock environment variables for testing.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    env_vars = {
        "OPENAI_API_KEY": "sk-test1234567890abcdefghijklmnopqrstuvwxyz",
        "LOG_LEVEL": "INFO",
        "JSON_LOGS": "false",
        "LOG_FILE_PATH": "logs/test.log",
        "ENABLE_FILE_LOGGING": "false"
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def clear_env_vars(monkeypatch):
    """Clears all environment variables for testing missing vars.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    env_vars_to_clear = [
        "OPENAI_API_KEY",
        "LOG_LEVEL",
        "JSON_LOGS",
        "LOG_FILE_PATH",
        "ENABLE_FILE_LOGGING"
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_structlog():
    """Provides a mocked structlog logger.
    
    Returns:
        Mock logger that doesn't actually log.
    """
    mock_logger = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.error = MagicMock()
    return mock_logger


@pytest.fixture(autouse=True)
def reset_structlog_context():
    """Automatically reset structlog context between tests.
    
    This prevents context pollution between test runs.
    """
    import structlog
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()
