"""Unit tests for config/env_validator.py module.

Tests environment variable validation, redaction functionality, error handling,
and configuration management for the application.
"""

import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from io import StringIO

from config.env_validator import (
    EnvVarConfig,
    EnvValidationError,
    EnvironmentValidator,
    validate_environment
)


class TestEnvVarConfig:
    """Test suite for EnvVarConfig dataclass."""

    def test_env_var_config_minimal(self):
        """Test creating EnvVarConfig with minimal required fields.
        
        Validates that only the name field is required and defaults are applied.
        """
        config = EnvVarConfig(name="TEST_VAR")
        
        assert config.name == "TEST_VAR"
        assert config.required is True  # Default
        assert config.default is None
        assert config.description == ""
        assert config.allowed_values is None
        assert config.validator is None

    def test_env_var_config_full(self):
        """Test creating EnvVarConfig with all fields specified.
        
        Validates that all configuration options can be set.
        """
        validator_func = lambda x: len(x) > 5
        
        config = EnvVarConfig(
            name="API_KEY",
            required=True,
            default="default_key",
            description="API key for external service",
            allowed_values=["key1", "key2"],
            validator=validator_func
        )
        
        assert config.name == "API_KEY"
        assert config.required is True
        assert config.default == "default_key"
        assert config.description == "API key for external service"
        assert config.allowed_values == ["key1", "key2"]
        assert config.validator == validator_func

    def test_env_var_config_optional_with_default(self):
        """Test creating optional EnvVarConfig with default value.
        
        Validates that optional variables can have default values.
        """
        config = EnvVarConfig(
            name="LOG_LEVEL",
            required=False,
            default="INFO"
        )
        
        assert config.required is False
        assert config.default == "INFO"


class TestEnvironmentValidator:
    """Test suite for EnvironmentValidator class."""

    def test_validator_initialization_strict_mode(self):
        """Test EnvironmentValidator initialization in strict mode.
        
        Validates that strict mode is enabled by default.
        """
        validator = EnvironmentValidator(strict_mode=True)
        
        assert validator.strict_mode is True
        assert validator.validation_errors == []
        assert validator.validation_warnings == []

    def test_validator_initialization_non_strict_mode(self):
        """Test EnvironmentValidator initialization in non-strict mode.
        
        Validates that strict mode can be disabled.
        """
        validator = EnvironmentValidator(strict_mode=False)
        
        assert validator.strict_mode is False

    def test_validate_all_success(self, mock_env_vars):
        """Test successful validation of all environment variables.
        
        Validates that all required env vars pass validation when properly set.
        """
        validator = EnvironmentValidator(strict_mode=True)
        
        with patch.object(validator, 'logger'):
            validated_vars = validator.validate_all()
        
        assert "OPENAI_API_KEY" in validated_vars
        assert validated_vars["OPENAI_API_KEY"].startswith("sk-")
        assert validated_vars["LOG_LEVEL"] == "INFO"
        assert len(validator.validation_errors) == 0

    def test_validate_all_missing_required_strict(self, clear_env_vars):
        """Test validation fails when required var is missing in strict mode.
        
        Validates that missing required variables cause validation to fail and exit.
        """
        validator = EnvironmentValidator(strict_mode=True)
        
        with patch.object(validator, 'logger'):
            with patch('sys.exit') as mock_exit:
                validator.validate_all()
                mock_exit.assert_called_once_with(1)
        
        assert len(validator.validation_errors) > 0
        assert any("OPENAI_API_KEY" in err for err in validator.validation_errors)

    def test_validate_all_missing_required_non_strict(self, clear_env_vars):
        """Test validation logs warnings when required var is missing in non-strict mode.
        
        Validates that non-strict mode logs warnings instead of exiting.
        """
        validator = EnvironmentValidator(strict_mode=False)
        
        with patch.object(validator, 'logger'):
            validated_vars = validator.validate_all()
        
        assert len(validator.validation_warnings) > 0

    def test_validate_single_required_present(self, mock_env_vars):
        """Test validating a single required environment variable that is present.
        
        Validates that a properly set required variable passes validation.
        """
        validator = EnvironmentValidator()
        config = EnvVarConfig(name="OPENAI_API_KEY", required=True)
        
        value = validator._validate_single(config)
        
        assert value is not None
        assert value.startswith("sk-")

    def test_validate_single_required_missing(self, clear_env_vars):
        """Test validating a single required environment variable that is missing.
        
        Validates that missing required variables raise EnvValidationError.
        """
        validator = EnvironmentValidator()
        config = EnvVarConfig(name="REQUIRED_VAR", required=True)
        
        with pytest.raises(EnvValidationError) as exc_info:
            validator._validate_single(config)
        
        assert "REQUIRED_VAR" in str(exc_info.value)
        assert "required" in str(exc_info.value).lower()

    def test_validate_single_optional_missing_with_default(self, clear_env_vars):
        """Test validating optional variable that is missing but has a default.
        
        Validates that default values are used when optional vars are not set.
        """
        validator = EnvironmentValidator()
        config = EnvVarConfig(
            name="OPTIONAL_VAR",
            required=False,
            default="default_value"
        )
        
        with patch.object(validator, 'logger'):
            value = validator._validate_single(config)
        
        assert value == "default_value"

    def test_validate_single_optional_missing_no_default(self, clear_env_vars):
        """Test validating optional variable that is missing with no default.
        
        Validates that optional vars without defaults return None when not set.
        """
        validator = EnvironmentValidator()
        config = EnvVarConfig(
            name="OPTIONAL_VAR",
            required=False,
            default=None
        )
        
        value = validator._validate_single(config)
        
        assert value is None

    def test_validate_single_allowed_values_valid(self, monkeypatch):
        """Test validation with allowed values when value is valid.
        
        Validates that values in the allowed list pass validation.
        """
        monkeypatch.setenv("TEST_VAR", "INFO")
        validator = EnvironmentValidator()
        config = EnvVarConfig(
            name="TEST_VAR",
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR"]
        )
        
        value = validator._validate_single(config)
        
        assert value == "INFO"

    def test_validate_single_allowed_values_invalid(self, monkeypatch):
        """Test validation with allowed values when value is invalid.
        
        Validates that values not in the allowed list raise EnvValidationError.
        """
        monkeypatch.setenv("TEST_VAR", "INVALID")
        validator = EnvironmentValidator()
        config = EnvVarConfig(
            name="TEST_VAR",
            allowed_values=["DEBUG", "INFO", "WARNING"]
        )
        
        with pytest.raises(EnvValidationError) as exc_info:
            validator._validate_single(config)
        
        assert "invalid" in str(exc_info.value).lower()
        assert "INVALID" in str(exc_info.value)

    def test_validate_single_custom_validator_pass(self, monkeypatch):
        """Test validation with custom validator function that passes.
        
        Validates that custom validator functions are executed correctly.
        """
        monkeypatch.setenv("API_KEY", "sk-1234567890abcdefghijklmnopqrstuvwxyz")
        validator = EnvironmentValidator()
        config = EnvVarConfig(
            name="API_KEY",
            validator=lambda x: len(x) > 20 and x.startswith("sk-")
        )
        
        value = validator._validate_single(config)
        
        assert value.startswith("sk-")
        assert len(value) > 20

    def test_validate_single_custom_validator_fail(self, monkeypatch):
        """Test validation with custom validator function that fails.
        
        Validates that failed custom validation raises EnvValidationError.
        """
        monkeypatch.setenv("API_KEY", "invalid_key")
        validator = EnvironmentValidator()
        config = EnvVarConfig(
            name="API_KEY",
            validator=lambda x: len(x) > 20 and x.startswith("sk-")
        )
        
        with pytest.raises(EnvValidationError) as exc_info:
            validator._validate_single(config)
        
        assert "validation" in str(exc_info.value).lower()

    def test_validate_single_custom_validator_exception(self, monkeypatch):
        """Test validation when custom validator raises an exception.
        
        Validates that exceptions in custom validators are caught and wrapped.
        """
        monkeypatch.setenv("TEST_VAR", "value")
        validator = EnvironmentValidator()
        
        def buggy_validator(x):
            raise ValueError("Validator error")
        
        config = EnvVarConfig(
            name="TEST_VAR",
            validator=buggy_validator
        )
        
        with pytest.raises(EnvValidationError) as exc_info:
            validator._validate_single(config)
        
        assert "validation error" in str(exc_info.value).lower()

    def test_validate_single_empty_string_treated_as_missing(self, monkeypatch):
        """Test that empty string values are treated as missing.
        
        Validates that whitespace-only values trigger missing variable logic.
        """
        monkeypatch.setenv("REQUIRED_VAR", "   ")
        validator = EnvironmentValidator()
        config = EnvVarConfig(name="REQUIRED_VAR", required=True)
        
        with pytest.raises(EnvValidationError) as exc_info:
            validator._validate_single(config)
        
        assert "required" in str(exc_info.value).lower()


class TestRedactionFunctionality:
    """Test suite for redaction functionality in EnvironmentValidator."""

    def test_should_redact_exact_match(self):
        """Test redaction with exact pattern match.
        
        Validates that exact variable names are matched for redaction.
        """
        validator = EnvironmentValidator()
        
        assert validator._should_redact("PASSWORD", ["PASSWORD"]) is True
        assert validator._should_redact("USERNAME", ["PASSWORD"]) is False

    def test_should_redact_suffix_wildcard(self):
        """Test redaction with suffix wildcard pattern (e.g., *_KEY).
        
        Validates that patterns ending with * match variable name suffixes.
        """
        validator = EnvironmentValidator()
        
        assert validator._should_redact("OPENAI_API_KEY", ["*_KEY"]) is True
        assert validator._should_redact("DATABASE_KEY", ["*_KEY"]) is True
        assert validator._should_redact("API_SECRET", ["*_KEY"]) is False

    def test_should_redact_prefix_wildcard(self):
        """Test redaction with prefix wildcard pattern (e.g., PASSWORD*).
        
        Validates that patterns starting with * match variable name prefixes.
        """
        validator = EnvironmentValidator()
        
        assert validator._should_redact("PASSWORD_HASH", ["PASSWORD*"]) is True
        assert validator._should_redact("PASSWORD", ["PASSWORD*"]) is True
        assert validator._should_redact("USER_PASSWORD", ["PASSWORD*"]) is False

    def test_should_redact_middle_wildcard(self):
        """Test redaction with wildcard in the middle (e.g., *API*KEY*).
        
        Validates that patterns with wildcards in the middle match correctly.
        """
        validator = EnvironmentValidator()
        
        assert validator._should_redact("OPENAI_API_KEY", ["*API*KEY*"]) is True
        assert validator._should_redact("MY_API_SECRET_KEY", ["*API*KEY*"]) is True
        assert validator._should_redact("DATABASE_PASSWORD", ["*API*KEY*"]) is False

    def test_should_redact_multiple_patterns(self):
        """Test redaction with multiple patterns.
        
        Validates that any matching pattern triggers redaction.
        """
        validator = EnvironmentValidator()
        patterns = ["*_KEY", "*_SECRET", "PASSWORD*"]
        
        assert validator._should_redact("API_KEY", patterns) is True
        assert validator._should_redact("CLIENT_SECRET", patterns) is True
        assert validator._should_redact("PASSWORD_HASH", patterns) is True
        assert validator._should_redact("USERNAME", patterns) is False

    def test_should_redact_case_sensitive(self):
        """Test that redaction pattern matching is case-sensitive.
        
        Validates that pattern matching respects case.
        """
        validator = EnvironmentValidator()
        
        assert validator._should_redact("API_KEY", ["*_KEY"]) is True
        assert validator._should_redact("api_key", ["*_KEY"]) is False
        assert validator._should_redact("API_key", ["*_KEY"]) is False

    def test_redact_value_non_empty(self):
        """Test redacting a non-empty value.
        
        Validates that sensitive values are fully redacted.
        """
        validator = EnvironmentValidator()
        
        redacted = validator._redact_value("sk-1234567890abcdefghijklmnopqrstuvwxyz")
        
        assert redacted == "***REDACTED***"
        assert "sk-" not in redacted
        assert "1234" not in redacted

    def test_redact_value_empty_string(self):
        """Test redacting an empty string.
        
        Validates that empty values are also redacted.
        """
        validator = EnvironmentValidator()
        
        redacted = validator._redact_value("")
        
        assert redacted == "***REDACTED***"

    def test_redact_value_none(self):
        """Test redacting a None value.
        
        Validates that None values are handled in redaction.
        """
        validator = EnvironmentValidator()
        
        redacted = validator._redact_value(None)
        
        assert redacted == "***REDACTED***"

    def test_print_env_summary_with_redaction(self, mock_env_vars, capsys):
        """Test printing environment summary with sensitive values redacted.
        
        Validates that sensitive values are redacted in the printed summary.
        """
        validator = EnvironmentValidator()
        
        with patch.object(validator, 'logger'):
            validated_vars = validator.validate_all()
        
        validator.print_env_summary(validated_vars, redact_patterns=["*_KEY"])
        
        captured = capsys.readouterr()
        
        assert "***REDACTED***" in captured.out
        assert "sk-" not in captured.out  # API key should be redacted
        assert "OPENAI_API_KEY" in captured.out  # Variable name shown

    def test_print_env_summary_without_redaction(self, mock_env_vars, capsys):
        """Test printing environment summary without redaction.
        
        Validates that values are shown when no redaction patterns match.
        """
        validator = EnvironmentValidator()
        
        with patch.object(validator, 'logger'):
            validated_vars = validator.validate_all()
        
        # Use patterns that don't match any variables
        validator.print_env_summary(validated_vars, redact_patterns=["NONEXISTENT*"])
        
        captured = capsys.readouterr()
        
        assert "INFO" in captured.out  # LOG_LEVEL should be visible
        assert "false" in captured.out  # JSON_LOGS should be visible

    def test_print_env_summary_default_redaction_patterns(self, mock_env_vars, capsys):
        """Test printing environment summary with default redaction patterns.
        
        Validates that default patterns (*_KEY, *_SECRET, etc.) are applied.
        """
        validator = EnvironmentValidator()
        
        with patch.object(validator, 'logger'):
            validated_vars = validator.validate_all()
        
        # Use default patterns (None)
        validator.print_env_summary(validated_vars, redact_patterns=None)
        
        captured = capsys.readouterr()
        
        assert "***REDACTED***" in captured.out
        assert "OPENAI_API_KEY" in captured.out


class TestValidateEnvironmentFunction:
    """Test suite for the validate_environment convenience function."""

    def test_validate_environment_success(self, mock_env_vars):
        """Test validate_environment function with valid environment.
        
        Validates that the convenience function returns validated vars.
        """
        with patch('config.env_validator.EnvironmentValidator') as MockValidator:
            mock_instance = MockValidator.return_value
            mock_instance.validate_all.return_value = {
                "OPENAI_API_KEY": "sk-test123",
                "LOG_LEVEL": "INFO"
            }
            mock_instance.print_env_summary = MagicMock()
            
            result = validate_environment(strict_mode=True, verbose=True)
            
            assert "OPENAI_API_KEY" in result
            MockValidator.assert_called_once_with(strict_mode=True)
            mock_instance.validate_all.assert_called_once()
            mock_instance.print_env_summary.assert_called_once()

    def test_validate_environment_non_verbose(self, mock_env_vars):
        """Test validate_environment function with verbose=False.
        
        Validates that summary is not printed when verbose is False.
        """
        with patch('config.env_validator.EnvironmentValidator') as MockValidator:
            mock_instance = MockValidator.return_value
            mock_instance.validate_all.return_value = {}
            mock_instance.print_env_summary = MagicMock()
            
            validate_environment(strict_mode=True, verbose=False)
            
            mock_instance.print_env_summary.assert_not_called()

    def test_validate_environment_custom_redaction_patterns(self, mock_env_vars):
        """Test validate_environment with custom redaction patterns.
        
        Validates that custom redaction patterns are passed to print_env_summary.
        """
        with patch('config.env_validator.EnvironmentValidator') as MockValidator:
            mock_instance = MockValidator.return_value
            mock_instance.validate_all.return_value = {}
            mock_instance.print_env_summary = MagicMock()
            
            custom_patterns = ["*CUSTOM*", "SECRET*"]
            validate_environment(
                strict_mode=True,
                verbose=True,
                redact_patterns=custom_patterns
            )
            
            mock_instance.print_env_summary.assert_called_once()
            call_args = mock_instance.print_env_summary.call_args
            assert call_args[1]["redact_patterns"] == custom_patterns

    def test_validate_environment_non_strict_mode(self, clear_env_vars):
        """Test validate_environment in non-strict mode with missing vars.
        
        Validates that non-strict mode doesn't exit on validation errors.
        """
        with patch('config.env_validator.EnvironmentValidator') as MockValidator:
            mock_instance = MockValidator.return_value
            mock_instance.validate_all.return_value = {}
            
            # Should not raise or exit
            result = validate_environment(strict_mode=False, verbose=False)
            
            MockValidator.assert_called_once_with(strict_mode=False)


class TestEnvValidationError:
    """Test suite for EnvValidationError exception."""

    def test_env_validation_error_creation(self):
        """Test creating EnvValidationError with a message.
        
        Validates that the custom exception can be created and raised.
        """
        error = EnvValidationError("Test error message")
        
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_env_validation_error_raised(self):
        """Test raising and catching EnvValidationError.
        
        Validates that the exception can be raised and caught properly.
        """
        with pytest.raises(EnvValidationError) as exc_info:
            raise EnvValidationError("Validation failed")
        
        assert "Validation failed" in str(exc_info.value)


class TestEdgeCases:
    """Test suite for edge cases and error scenarios."""

    def test_validate_with_special_characters_in_value(self, monkeypatch):
        """Test validation with special characters in environment variable value.
        
        Validates that special characters are handled correctly.
        """
        special_value = "key!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        monkeypatch.setenv("SPECIAL_VAR", special_value)
        
        validator = EnvironmentValidator()
        config = EnvVarConfig(name="SPECIAL_VAR", required=True)
        
        value = validator._validate_single(config)
        
        assert value == special_value

    def test_validate_with_unicode_in_value(self, monkeypatch):
        """Test validation with unicode characters in environment variable value.
        
        Validates that unicode characters are preserved.
        """
        unicode_value = "你好世界🌍"
        monkeypatch.setenv("UNICODE_VAR", unicode_value)
        
        validator = EnvironmentValidator()
        config = EnvVarConfig(name="UNICODE_VAR", required=True)
        
        value = validator._validate_single(config)
        
        assert value == unicode_value

    def test_validate_with_very_long_value(self, monkeypatch):
        """Test validation with very long environment variable value.
        
        Validates that long values are handled without issues.
        """
        long_value = "x" * 10000
        monkeypatch.setenv("LONG_VAR", long_value)
        
        validator = EnvironmentValidator()
        config = EnvVarConfig(name="LONG_VAR", required=True)
        
        value = validator._validate_single(config)
        
        assert len(value) == 10000

    def test_validate_with_newlines_in_value(self, monkeypatch):
        """Test validation with newlines in environment variable value.
        
        Validates that multiline values are preserved.
        """
        multiline_value = "line1\nline2\nline3"
        monkeypatch.setenv("MULTILINE_VAR", multiline_value)
        
        validator = EnvironmentValidator()
        config = EnvVarConfig(name="MULTILINE_VAR", required=True)
        
        value = validator._validate_single(config)
        
        assert "\n" in value
        assert value == multiline_value

    def test_multiple_validation_errors_accumulated(self, clear_env_vars):
        """Test that multiple validation errors are accumulated.
        
        Validates that all validation errors are collected and reported.
        """
        validator = EnvironmentValidator(strict_mode=True)
        
        with patch.object(validator, 'logger'):
            with patch('sys.exit'):
                validator.validate_all()
        
        # Should have error for missing OPENAI_API_KEY at minimum
        assert len(validator.validation_errors) >= 1
        assert any("OPENAI_API_KEY" in err for err in validator.validation_errors)
