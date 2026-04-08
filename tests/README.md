# Email Handler LangGraph - Test Suite

Comprehensive unit tests for the email_handler-langgraph application.

## Overview

This test suite provides extensive coverage of critical components:

- **agents/handler.py** - Email handling logic, workflow transitions, and state management
- **agents/state.py** - State structure and data validation
- **config/env_validator.py** - Environment validation, redaction, and error handling
- **main.py** - Application initialization and startup flow

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── test_state.py            # Tests for state TypedDict structures
├── test_env_validator.py    # Tests for environment validation
├── test_handler.py          # Tests for workflow nodes and graph
├── test_main.py             # Tests for application initialization
└── README.md                # This file
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_handler.py
```

### Run specific test class
```bash
pytest tests/test_handler.py::TestClassifyIntent
```

### Run specific test function
```bash
pytest tests/test_handler.py::TestClassifyIntent::test_classify_intent_billing_routes_to_human_review
```

### Run with coverage report
```bash
pytest --cov=agents --cov=config --cov=main --cov-report=html
```

### Run only unit tests
```bash
pytest -m unit
```

### Run with verbose output
```bash
pytest -v
```

### Run with detailed output for failures
```bash
pytest -vv --tb=long
```

## Test Coverage

### agents/state.py
- ✅ EmailClassification structure validation
- ✅ All intent literal values (question, bug, billing, feature, complex)
- ✅ All urgency literal values (low, medium, high, critical)
- ✅ EmailAgentState structure validation
- ✅ State with classification, search results, customer history
- ✅ Edge cases: empty strings, special characters, long content

### config/env_validator.py
- ✅ EnvVarConfig dataclass creation
- ✅ Environment validation (strict and non-strict modes)
- ✅ Required vs optional variables
- ✅ Default value handling
- ✅ Allowed values validation
- ✅ Custom validator functions
- ✅ Redaction patterns (exact, wildcard, prefix, suffix)
- ✅ Error accumulation and reporting
- ✅ Edge cases: special characters, unicode, long values

### agents/handler.py
- ✅ read_email: Email content extraction and message creation
- ✅ classify_intent: LLM classification and routing logic
  - Billing → human_review
  - Critical urgency → human_review
  - Question/Feature → search_documentation
  - Bug → bug_tracking
  - Complex → draft_response
- ✅ search_documentation: Search execution and result handling
- ✅ bug_tracking: Ticket creation and routing
- ✅ draft_response: Response generation and routing based on urgency
- ✅ human_review: Interrupt handling, approval/rejection routing
- ✅ send_reply: Email sending (terminal node)
- ✅ get_graph: Graph construction, retry policies, checkpointer
- ✅ Error handling for all nodes
- ✅ State preservation and updates

### main.py
- ✅ Correlation ID generation
- ✅ Initial state creation
- ✅ Graph invocation with configuration
- ✅ Human review interrupt handling
- ✅ Workflow resume with human response
- ✅ Logging at all workflow stages
- ✅ Error handling (graph invocation, resume, visualization)
- ✅ Graph visualization generation
- ✅ Module initialization (dotenv, validation, logging)

## Fixtures

### State Fixtures (conftest.py)
- `sample_email_state` - Basic email state with minimal fields
- `urgent_billing_state` - Urgent billing issue state
- `bug_report_state` - Bug report state
- `question_state` - Customer question state with search results
- `sample_classification` - Sample EmailClassification

### Mock Fixtures (conftest.py)
- `mock_llm` - Mocked ChatOpenAI instance
- `mock_structured_llm` - Mocked structured LLM for classification
- `mock_env_vars` - Sets up test environment variables
- `clear_env_vars` - Clears all environment variables
- `mock_structlog` - Mocked structlog logger

### Auto-use Fixtures
- `reset_structlog_context` - Automatically resets structlog context between tests

## Test Principles

### Isolation
- All tests are isolated and independent
- No tests depend on actual environment variables
- External dependencies (OpenAI API, file I/O) are mocked
- Structlog context is reset between tests

### Coverage
- Happy paths and success scenarios
- Edge cases (empty values, special characters, long content)
- Error scenarios and exception handling
- Both success and failure paths

### Mocking Strategy
- Mock external APIs (OpenAI, email services)
- Mock file I/O operations
- Mock environment variables for testing
- Mock logging to verify log calls without output

### Documentation
- Every test has a docstring explaining:
  - What the test validates
  - The scenario being tested
  - Expected behavior

## Coverage Goals

- **Target**: >90% code coverage for critical business logic
- **Focus Areas**:
  - Workflow routing logic
  - State management
  - Error handling
  - Environment validation
  - Classification and response generation

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: pytest --cov --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Troubleshooting

### Import Errors
If you encounter import errors, ensure the parent directory is in the Python path:
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

### Environment Variable Conflicts
Tests use `monkeypatch` to set/clear environment variables. If you see conflicts:
- Ensure `clear_env_vars` fixture is used when testing missing vars
- Check that `reset_structlog_context` is working (it's auto-use)

### Mock Not Working
If mocks aren't being applied:
- Verify the patch path matches the import path in the module under test
- Use `patch('agents.handler.llm')` not `patch('langchain_openai.ChatOpenAI')`
- Ensure patches are applied before the function is called

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_<function>_<scenario>`
2. **Add docstrings**: Explain what the test validates
3. **Use fixtures**: Reuse existing fixtures from conftest.py
4. **Test edge cases**: Don't just test happy paths
5. **Mock external dependencies**: Never call real APIs in tests
6. **Keep tests focused**: One behavior per test
7. **Update this README**: Document new test coverage

## License

Same as parent project.
