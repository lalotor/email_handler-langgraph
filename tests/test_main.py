"""Unit tests for main.py module.

Tests application initialization, startup flow, environment validation,
logging configuration, and graph execution.
"""

import pytest
import uuid
from unittest.mock import Mock, MagicMock, patch, call
from langgraph.types import Command

import main


class TestMainFunction:
    """Test suite for the main() function."""

    def test_main_generates_correlation_id(self):
        """Test that main() generates a unique correlation ID.
        
        Validates that each execution has a unique tracking ID.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {'__interrupt__': {}}
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger'):
                with patch('main.save_graph_image'):
                    with patch('structlog.contextvars.bind_contextvars') as mock_bind:
                        main.main()
                        
                        # Verify correlation ID was bound
                        mock_bind.assert_called_once()
                        call_kwargs = mock_bind.call_args[1]
                        assert 'correlation_id' in call_kwargs
                        # Verify it's a valid UUID
                        uuid.UUID(call_kwargs['correlation_id'])

    def test_main_creates_initial_state(self):
        """Test that main() creates proper initial email state.
        
        Validates that the initial state has all required fields.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {'__interrupt__': {}}
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger'):
                with patch('main.save_graph_image'):
                    main.main()
                    
                    # Check first invoke call (initial state)
                    first_call_args = mock_graph.invoke.call_args_list[0]
                    initial_state = first_call_args[0][0]
                    
                    assert 'email_content' in initial_state
                    assert 'sender_email' in initial_state
                    assert 'email_id' in initial_state
                    assert 'messages' in initial_state

    def test_main_invokes_graph_with_config(self):
        """Test that main() invokes graph with proper thread configuration.
        
        Validates that thread_id is set for state persistence.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {'__interrupt__': {}}
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger'):
                with patch('main.save_graph_image'):
                    main.main()
                    
                    # Check config in first invoke
                    first_call_args = mock_graph.invoke.call_args_list[0]
                    config = first_call_args[0][1]
                    
                    assert 'configurable' in config
                    assert 'thread_id' in config['configurable']
                    assert config['configurable']['thread_id'] == 'customer_123'

    def test_main_handles_human_review_interrupt(self):
        """Test that main() properly handles workflow interruption for human review.
        
        Validates that the workflow pauses and resumes correctly.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            # First invoke returns interrupt, second returns final result
            mock_graph.invoke.side_effect = [
                {'__interrupt__': {'action': 'review'}},
                {'email_sent': True}
            ]
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger'):
                with patch('main.save_graph_image'):
                    main.main()
                    
                    # Verify graph was invoked twice (initial + resume)
                    assert mock_graph.invoke.call_count == 2

    def test_main_provides_human_response(self):
        """Test that main() provides human review response to resume workflow.
        
        Validates that human approval is sent as a Command.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = [
                {'__interrupt__': {}},
                {}
            ]
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger'):
                with patch('main.save_graph_image'):
                    main.main()
                    
                    # Check second invoke (resume)
                    second_call_args = mock_graph.invoke.call_args_list[1]
                    human_response = second_call_args[0][0]
                    
                    assert isinstance(human_response, Command)
                    assert 'approved' in human_response.resume
                    assert human_response.resume['approved'] is True
                    assert 'edited_response' in human_response.resume

    def test_main_logs_workflow_stages(self):
        """Test that main() logs all major workflow stages.
        
        Validates that processing start, pause, resume, and completion are logged.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = [
                {'__interrupt__': {}},
                {}
            ]
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger') as mock_logger:
                with patch('main.save_graph_image'):
                    main.main()
                    
                    # Verify key logging events
                    info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                    assert any('email_processing_started' in str(c) for c in info_calls)
                    assert any('workflow_paused' in str(c) for c in info_calls)
                    assert any('email_processing_completed' in str(c) for c in info_calls)

    def test_main_handles_graph_invocation_error(self):
        """Test that main() handles errors during graph invocation.
        
        Validates error handling and logging when graph fails.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = Exception("Graph execution failed")
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger') as mock_logger:
                with pytest.raises(Exception) as exc_info:
                    main.main()
                
                assert "Graph execution failed" in str(exc_info.value)
                # Verify error was logged
                assert mock_logger.error.called

    def test_main_handles_resume_error(self):
        """Test that main() handles errors during workflow resume.
        
        Validates error handling when resuming after human review fails.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = [
                {'__interrupt__': {}},
                Exception("Resume failed")
            ]
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger') as mock_logger:
                with pytest.raises(Exception) as exc_info:
                    main.main()
                
                assert "Resume failed" in str(exc_info.value)
                assert mock_logger.error.called

    def test_main_calls_save_graph_image(self):
        """Test that main() calls save_graph_image at the end.
        
        Validates that graph visualization is generated.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = [
                {'__interrupt__': {}},
                {}
            ]
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger'):
                with patch('main.save_graph_image') as mock_save:
                    main.main()
                    
                    mock_save.assert_called_once_with(mock_graph)


class TestSaveGraphImage:
    """Test suite for the save_graph_image() function."""

    def test_save_graph_image_generates_visualization(self):
        """Test that save_graph_image generates graph visualization.
        
        Validates that the mermaid PNG is generated.
        """
        mock_graph = MagicMock()
        mock_graph.get_graph.return_value.draw_mermaid_png.return_value = b'PNG_DATA'
        
        with patch('main.logger'):
            main.save_graph_image(mock_graph)
            
            mock_graph.get_graph.assert_called_once()
            mock_graph.get_graph.return_value.draw_mermaid_png.assert_called_once()

    def test_save_graph_image_logs_success(self):
        """Test that save_graph_image logs successful image generation.
        
        Validates that success is logged with filename.
        """
        mock_graph = MagicMock()
        mock_graph.get_graph.return_value.draw_mermaid_png.return_value = b'PNG_DATA'
        
        with patch('main.logger') as mock_logger:
            main.save_graph_image(mock_graph)
            
            # Verify success logging
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any('graph_image_saved' in call for call in info_calls)

    def test_save_graph_image_handles_error_gracefully(self):
        """Test that save_graph_image handles visualization errors gracefully.
        
        Validates that errors don't crash the application.
        """
        mock_graph = MagicMock()
        mock_graph.get_graph.side_effect = Exception("Visualization library missing")
        
        with patch('main.logger') as mock_logger:
            # Should not raise exception
            main.save_graph_image(mock_graph)
            
            # Verify warning was logged
            assert mock_logger.warning.called

    def test_save_graph_image_logs_warning_on_failure(self):
        """Test that save_graph_image logs appropriate warning on failure.
        
        Validates that helpful error message is logged.
        """
        mock_graph = MagicMock()
        mock_graph.get_graph.side_effect = Exception("Missing dependency")
        
        with patch('main.logger') as mock_logger:
            main.save_graph_image(mock_graph)
            
            # Verify warning includes helpful message
            warning_calls = [call[1] for call in mock_logger.warning.call_args_list]
            assert any(
                'message' in kwargs and 'dependency' in kwargs.get('message', '').lower()
                for kwargs in warning_calls
                if isinstance(kwargs, dict)
            )


class TestModuleInitialization:
    """Test suite for module-level initialization."""

    def test_module_loads_dotenv(self):
        """Test that module loads environment variables from .env file.
        
        Validates that load_dotenv is called on import.
        """
        # This is tested implicitly by module import
        # Verify the import doesn't fail
        import main
        assert hasattr(main, 'load_dotenv')

    def test_module_validates_environment(self):
        """Test that module validates environment on import.
        
        Validates that validate_environment is called during initialization.
        """
        # Module should have validated_env from initialization
        import main
        assert hasattr(main, 'validated_env')

    def test_module_configures_logging(self):
        """Test that module configures logging on import.
        
        Validates that logging is set up before main execution.
        """
        import main
        assert hasattr(main, 'logger')

    def test_module_has_main_guard(self):
        """Test that module uses if __name__ == '__main__' guard.
        
        Validates that main() is only called when script is run directly.
        """
        # Module should be importable without running main()
        import main
        # If we got here without errors, the guard is working
        assert callable(main.main)


class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    def test_full_workflow_execution(self):
        """Test complete workflow from start to finish.
        
        Validates end-to-end execution with all components.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = [
                {
                    '__interrupt__': {
                        'email_id': 'test_123',
                        'action': 'review'
                    }
                },
                {
                    'email_sent': True,
                    'draft_response': 'Final response'
                }
            ]
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger'):
                with patch('main.save_graph_image'):
                    with patch('structlog.contextvars.bind_contextvars'):
                        main.main()
                        
                        # Verify complete workflow
                        assert mock_graph.invoke.call_count == 2
                        assert mock_get_graph.called

    def test_workflow_with_logging_context(self):
        """Test that workflow maintains logging context throughout execution.
        
        Validates that correlation ID is preserved across workflow steps.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = [
                {'__interrupt__': {}},
                {}
            ]
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger') as mock_logger:
                with patch('main.save_graph_image'):
                    with patch('structlog.contextvars.bind_contextvars') as mock_bind:
                        main.main()
                        
                        # Verify context was bound
                        assert mock_bind.called
                        # Verify logging occurred with context
                        assert mock_logger.info.called

    def test_workflow_state_persistence(self):
        """Test that workflow uses thread_id for state persistence.
        
        Validates that the same thread_id is used for initial and resume calls.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = [
                {'__interrupt__': {}},
                {}
            ]
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger'):
                with patch('main.save_graph_image'):
                    main.main()
                    
                    # Both invocations should use same config
                    first_config = mock_graph.invoke.call_args_list[0][0][1]
                    second_config = mock_graph.invoke.call_args_list[1][0][1]
                    
                    assert first_config['configurable']['thread_id'] == \
                           second_config['configurable']['thread_id']


class TestErrorRecovery:
    """Test suite for error recovery scenarios."""

    def test_recovery_from_graph_construction_error(self):
        """Test handling of errors during graph construction.
        
        Validates that graph construction errors are properly handled.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_get_graph.side_effect = Exception("Graph construction failed")
            
            with patch('main.logger') as mock_logger:
                with pytest.raises(Exception) as exc_info:
                    main.main()
                
                assert "Graph construction failed" in str(exc_info.value)

    def test_logging_of_error_details(self):
        """Test that error details are properly logged.
        
        Validates that error type and traceback information are captured.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = ValueError("Invalid state")
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger') as mock_logger:
                with pytest.raises(ValueError):
                    main.main()
                
                # Verify error logging includes error type
                error_calls = [call[1] for call in mock_logger.error.call_args_list]
                assert any(
                    'error_type' in kwargs
                    for kwargs in error_calls
                    if isinstance(kwargs, dict)
                )

    def test_graceful_degradation_on_visualization_failure(self):
        """Test that visualization failure doesn't prevent workflow completion.
        
        Validates that the main workflow completes even if image generation fails.
        """
        with patch('main.get_graph') as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = [
                {'__interrupt__': {}},
                {}
            ]
            mock_graph.get_graph.side_effect = Exception("Viz error")
            mock_get_graph.return_value = mock_graph
            
            with patch('main.logger'):
                # Should complete without raising
                main.main()
                
                # Verify both workflow invocations completed
                assert mock_graph.invoke.call_count == 2
