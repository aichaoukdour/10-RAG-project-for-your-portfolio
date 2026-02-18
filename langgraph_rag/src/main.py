import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import get_graph

def main():
    print("🕸️ Initializing LangGraph Retrieval Agent...")
    try:
        graph = get_graph()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    print("\n" + "="*50)
    print("      🕸️ LANGGRAPH AGENTIC RAG SYSTEM")
    print("="*50)
    print("Target Knowledge: Lilian Weng's Blog (Reward Hacking, Hallucination, Video Diffusion)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Goodbye!")
                break
            
            print("\n🚀 Executing Agent Graph...")
            
            # Stream the graph execution to show node updates
            config = {"configurable": {"thread_id": "1"}}
            inputs = {"messages": [{"role": "user", "content": user_input}]}
            
            for output in graph.stream(inputs, config=config):
                for node_name, state_update in output.items():
                    print(f"\n--- Node: {node_name} ---")
                    # If the update has messages, print the last one (ai message or tool output)
                    if "messages" in state_update:
                        last_msg = state_update["messages"][-1]
                        if hasattr(last_msg, "content"):
                            print(f"{last_msg.content}")
                        else:
                            # For ToolMessages or others that might not have .content easily readable
                            print(f"{last_msg}")
            
            print("\n" + "="*50)

        except KeyboardInterrupt:
            print("\n🤖 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
