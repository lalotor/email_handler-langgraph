from langgraph.types import Command
from agents.handler import get_graph

def main():
    # Test with an urgent billing issue
    initial_state = {
        "email_content": "I was charged twice for my subscription! This is urgent!",
        "sender_email": "customer@example.com",
        "email_id": "email_123",
        "messages": []
    }

    # Run with a thread_id for persistence
    config = {"configurable": {"thread_id": "customer_123"}}
    graph = get_graph()
    result = graph.invoke(initial_state, config)
    # The graph will pause at human_review
    print(f"human review interrupt:{result['__interrupt__']}")

    # When ready, provide human input to resume
    human_response = Command(
        resume={
            "approved": True,
            "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund..."
        }
    )

    # Resume execution
    final_result = graph.invoke(human_response, config)
    print(f"Email sent successfully!")

    save_graph_image(graph)

def save_graph_image(graph):
    try:
        # Generate the image data
        graph_image = graph.get_graph().draw_mermaid_png()

        # Save the image to a file
        with open("graph_image.png", "wb") as f:
            f.write(graph_image)
        # logger.info("Graph image saved as graph_image.png")
    except Exception as e:
        # logger.error(f"Error displaying image: {e}", exc_info=True)
        pass  # Handle the optional dependencies for image rendering

if __name__ == "__main__":
    main()
