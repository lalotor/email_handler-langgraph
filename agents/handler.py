from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from agents.state import EmailAgentState, EmailClassification
import structlog

# Load variables from .env file
load_dotenv()

# Get logger for this module
logger = structlog.get_logger(__name__)

llm = ChatOpenAI(model="gpt-5-nano")

def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content"""
    logger.info(
        "node_started",
        node="read_email",
        email_id=state.get('email_id'),
        sender=state.get('sender_email')
    )
    
    # In production, this would connect to your email service
    logger.debug(
        "email_content_extracted",
        content_length=len(state.get('email_content', '')),
        has_sender=bool(state.get('sender_email'))
    )
    
    logger.info(
        "node_completed",
        node="read_email",
        email_id=state.get('email_id')
    )
    
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """Use LLM to classify email intent and urgency, then route accordingly"""
    logger.info(
        "node_started",
        node="classify_intent",
        email_id=state.get('email_id')
    )

    # Create structured LLM that returns EmailClassification dict
    structured_llm = llm.with_structured_output(EmailClassification)

    # Format the prompt on-demand, not stored in state
    classification_prompt = f"""
    Analyze this customer email and classify it:

    Email: {state['email_content']}
    From: {state['sender_email']}

    Provide classification including intent, urgency, topic, and summary.
    """

    logger.debug(
        "llm_classification_started",
        prompt_length=len(classification_prompt)
    )

    try:
        # Get structured response directly as dict
        classification = structured_llm.invoke(classification_prompt)
        
        logger.info(
            "email_classified",
            email_id=state.get('email_id'),
            intent=classification['intent'],
            urgency=classification['urgency'],
            topic=classification['topic'],
            summary=classification.get('summary', '')[:100]  # Truncate for logging
        )
    except Exception as e:
        logger.error(
            "classification_failed",
            email_id=state.get('email_id'),
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise

    # Determine next node based on classification
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    logger.info(
        "routing_decision",
        node="classify_intent",
        email_id=state.get('email_id'),
        next_node=goto,
        reason=f"intent={classification['intent']}, urgency={classification['urgency']}"
    )

    # Store classification as a single dict in state
    return Command(
        update={"classification": classification},
        goto=goto
    )

def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search knowledge base for relevant information"""
    logger.info(
        "node_started",
        node="search_documentation",
        email_id=state.get('email_id')
    )

    # Build search query from classification
    classification = state.get('classification') or {}
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"
    
    logger.debug(
        "documentation_search_query",
        query=query,
        intent=classification.get('intent'),
        topic=classification.get('topic')
    )

    try:
        # Implement your search logic here
        # Store raw search results, not formatted text
        search_results = [
            "Reset password via Settings > Security > Change Password",
            "Password must be at least 12 characters",
            "Include uppercase, lowercase, numbers, and symbols"
        ]
        
        logger.info(
            "documentation_search_completed",
            email_id=state.get('email_id'),
            results_count=len(search_results),
            query=query
        )
    # except SearchAPIError as e:
    except Exception as e:
        # For recoverable search errors, store error and continue
        logger.warning(
            "documentation_search_failed",
            email_id=state.get('email_id'),
            query=query,
            error=str(e),
            error_type=type(e).__name__,
            fallback="Continuing with error message in results"
        )
        search_results = [f"Search temporarily unavailable: {str(e)}"]

    logger.info(
        "node_completed",
        node="search_documentation",
        email_id=state.get('email_id'),
        next_node="draft_response"
    )

    return Command(
        update={"search_results": search_results},  # Store raw results or error
        goto="draft_response"
    )

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Create or update bug tracking ticket"""
    logger.info(
        "node_started",
        node="bug_tracking",
        email_id=state.get('email_id')
    )
    
    classification = state.get('classification', {})
    logger.debug(
        "creating_bug_ticket",
        email_id=state.get('email_id'),
        topic=classification.get('topic'),
        urgency=classification.get('urgency')
    )

    # Create ticket in your bug tracking system
    ticket_id = "BUG-12345"  # Would be created via API
    
    logger.info(
        "bug_ticket_created",
        email_id=state.get('email_id'),
        ticket_id=ticket_id,
        next_node="draft_response"
    )

    return Command(
        update={
            "search_results": [f"Bug ticket {ticket_id} created"],
            "current_step": "bug_tracked"
        },
        goto="draft_response"
    )

def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Generate response using context and route based on quality"""
    logger.info(
        "node_started",
        node="draft_response",
        email_id=state.get('email_id')
    )

    classification = state.get('classification', {})

    # Format context from raw state data on-demand
    context_sections = []

    if state.get('search_results'):
        # Format search results for the prompt
        formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")
        logger.debug(
            "context_added",
            context_type="search_results",
            results_count=len(state['search_results'])
        )

    if state.get('customer_history'):
        # Format customer data for the prompt
        context_sections.append(f"Customer tier: {state['customer_history'].get('tier', 'standard')}")
        logger.debug(
            "context_added",
            context_type="customer_history",
            tier=state['customer_history'].get('tier')
        )

    # Build the prompt with formatted context
    draft_prompt = f"""
    Draft a response to this customer email:
    {state['email_content']}

    Email intent: {classification.get('intent', 'unknown')}
    Urgency level: {classification.get('urgency', 'medium')}

    {chr(10).join(context_sections)}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    """

    logger.debug(
        "llm_draft_generation_started",
        email_id=state.get('email_id'),
        prompt_length=len(draft_prompt),
        context_sections_count=len(context_sections)
    )

    try:
        response = llm.invoke(draft_prompt)
        
        logger.info(
            "draft_response_generated",
            email_id=state.get('email_id'),
            response_length=len(response.content),
            intent=classification.get('intent'),
            urgency=classification.get('urgency')
        )
    except Exception as e:
        logger.error(
            "draft_generation_failed",
            email_id=state.get('email_id'),
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise

    # Determine if human review needed based on urgency and intent
    needs_review = (
        classification.get('urgency') in ['high', 'critical'] or
        classification.get('intent') == 'complex'
    )

    # Route to appropriate next node
    goto = "human_review" if needs_review else "send_reply"
    
    logger.info(
        "routing_decision",
        node="draft_response",
        email_id=state.get('email_id'),
        next_node=goto,
        needs_review=needs_review,
        reason=f"urgency={classification.get('urgency')}, intent={classification.get('intent')}"
    )

    return Command(
        update={"draft_response": response.content},  # Store only the raw response
        goto=goto
    )

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""
    logger.info(
        "node_started",
        node="human_review",
        email_id=state.get('email_id')
    )

    classification = state.get('classification', {})
    
    logger.info(
        "workflow_paused_for_review",
        email_id=state.get('email_id'),
        urgency=classification.get('urgency'),
        intent=classification.get('intent'),
        draft_length=len(state.get('draft_response', ''))
    )

    # interrupt() must come first - any code before it will re-run on resume
    human_decision = interrupt({
        "email_id": state.get('email_id',''),
        "original_email": state.get('email_content',''),
        "draft_response": state.get('draft_response',''),
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "Please review and approve/edit this response"
    })

    # Now process the human's decision
    logger.info(
        "human_decision_received",
        email_id=state.get('email_id'),
        approved=human_decision.get("approved"),
        response_edited=human_decision.get("edited_response") != state.get('draft_response')
    )

    if human_decision.get("approved"):
        logger.info(
            "routing_decision",
            node="human_review",
            email_id=state.get('email_id'),
            next_node="send_reply",
            reason="Human approved response"
        )
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response',''))},
            goto="send_reply"
        )
    else:
        # Rejection means human will handle directly
        logger.info(
            "routing_decision",
            node="human_review",
            email_id=state.get('email_id'),
            next_node="END",
            reason="Human rejected - will handle manually"
        )
        return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> dict:
    """Send the email response"""
    logger.info(
        "node_started",
        node="send_reply",
        email_id=state.get('email_id')
    )
    
    draft_response = state.get('draft_response') or ''
    
    logger.debug(
        "sending_email",
        email_id=state.get('email_id'),
        recipient=state.get('sender_email'),
        response_length=len(draft_response)
    )
    
    try:
        # Integrate with email service
        logger.info(
            "sending_reply_preview",
            email_id=state.get('email_id'),
            reply_preview=draft_response[:100] + "..." if len(draft_response) > 100 else draft_response
        )
        
        logger.info(
            "email_sent_successfully",
            email_id=state.get('email_id'),
            recipient=state.get('sender_email'),
            response_preview=draft_response[:50] + "..." if len(draft_response) > 50 else draft_response
        )
    except Exception as e:
        logger.error(
            "email_send_failed",
            email_id=state.get('email_id'),
            recipient=state.get('sender_email'),
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise
    
    logger.info(
        "node_completed",
        node="send_reply",
        email_id=state.get('email_id')
    )
    
    return {}

def get_graph():
    logger.info("graph_construction_started")
    
    # Create the graph
    workflow = StateGraph(EmailAgentState)

    # Add nodes with appropriate error handling
    logger.debug("adding_workflow_nodes")
    workflow.add_node("read_email", read_email)
    workflow.add_node("classify_intent", classify_intent)

    # Add retry policy for nodes that might have transient failures
    workflow.add_node(
        "search_documentation",
        search_documentation,
        retry_policy=RetryPolicy(max_attempts=3)
    )
    logger.debug(
        "retry_policy_configured",
        node="search_documentation",
        max_attempts=3
    )
    
    workflow.add_node("bug_tracking", bug_tracking)
    workflow.add_node("draft_response", draft_response)
    workflow.add_node("human_review", human_review)
    workflow.add_node("send_reply", send_reply)

    # Add only the essential edges
    logger.debug("adding_workflow_edges")
    workflow.add_edge(START, "read_email")
    workflow.add_edge("read_email", "classify_intent")
    workflow.add_edge("send_reply", END)

    # Compile with checkpointer for persistence, in case run graph with Local_Server --> Please compile without checkpointer
    logger.debug("compiling_graph_with_checkpointer")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    logger.info(
        "graph_construction_completed",
        nodes_count=7,
        has_checkpointer=True
    )

    return app
