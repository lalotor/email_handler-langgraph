import uuid
import os
import structlog
from langgraph.types import Command
from dotenv import load_dotenv
from agents.handler import get_graph
from config.logging_config import configure_logging
from config.env_validator import validate_environment

# Load environment variables
load_dotenv()

# Validate environment variables before proceeding
# This ensures all required configuration is present
validated_env = validate_environment(verbose=True)

# Configure logging from environment variables
configure_logging(
    log_level=validated_env.get("LOG_LEVEL", "INFO"),
    json_logs=validated_env.get("JSON_LOGS", "false").lower() == "true",
    enable_file_logging=validated_env.get("ENABLE_FILE_LOGGING", "false").lower() == "true",
    log_file_path=validated_env.get("LOG_FILE_PATH", "logs/app.log")
)

logger = structlog.get_logger(__name__)

def main():
    # Generate correlation ID for this email processing session
    correlation_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    
    logger.info(
        "email_processing_started",
        correlation_id=correlation_id,
        workflow="email_handler"
    )
    
    # Test with an urgent billing issue
    initial_state = {
        "email_content": "I was charged twice for my subscription! This is urgent!",
        "sender_email": "customer@example.com",
        "email_id": "email_123",
        "messages": []
    }
    
    logger.debug(
        "initial_state_created",
        email_id=initial_state["email_id"],
        sender=initial_state["sender_email"],
        content_length=len(initial_state["email_content"])
    )

    # Run with a thread_id for persistence
    config = {"configurable": {"thread_id": "customer_123"}}




    
    logger.info(
        "graph_initialization",
        thread_id=config["configurable"]["thread_id"]
    )
    
    try:
        graph = get_graph()
        logger.debug("graph_compiled_successfully")
        
        result = graph.invoke(initial_state, config)
        
        # The graph will pause at human_review
        logger.info(
            "workflow_paused_for_human_review",
            interrupt_data=result.get('__interrupt__', {}),
            email_id=initial_state["email_id"]
        )
        logger.info(
            "human_review_interrupt_details",
            interrupt=result['__interrupt__']
        )
    except Exception as e:
        logger.error(
            "graph_invocation_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise

    # When ready, provide human input to resume
    human_response = Command(
        resume={
            "approved": True,
            "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund..."
        }
    )
    
    logger.info(
        "human_review_completed",
        approved=human_response.resume["approved"],
        response_edited=True
    )

    # Resume execution


    try:
        final_result = graph.invoke(human_response, config)
        logger.info(
            "email_processing_completed",
            email_id=initial_state["email_id"],
            final_state_keys=list(final_result.keys()) if isinstance(final_result, dict) else None
        )
        logger.info("email_sent_successfully_message")
    except Exception as e:
        logger.error(
            "workflow_resume_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise

    save_graph_image(graph)

def save_graph_image(graph):
    try:
        logger.debug("generating_graph_visualization")
        # Generate the image data
        graph_image = graph.get_graph().draw_mermaid_png()

        # Save the image to a file

        logger.info("graph_image_saved", filename="graph_image.png")
    except Exception as e:


        logger.warning(
            "graph_visualization_failed",
            error=str(e),
            error_type=type(e).__name__,
            message="Optional dependency for image rendering may be missing"
        )

if __name__ == "__main__":
    main()
