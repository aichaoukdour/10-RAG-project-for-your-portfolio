import sys
import os

# Add project root to sys.path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assistant import get_assistant_chain

def main():
    print("🤖 Initializing Real-Time AI Assistant...")
    try:
        chain = get_assistant_chain()
    except Exception as e:
        print(f"❌ Failed to initialize chain: {e}")
        print("💡 Make sure Ollama is running and the model is pulled.")
        return

    print("\n" + "="*50)
    print("      🤖 LANGCHAIN REAL-TIME ASSISTANT")
    print("      (Powered by Ollama + DuckDuckGo)")
    print("="*50)
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_query = input("\nYou: ").strip()
            if not user_query:
                continue
            
            if user_query.lower() in ["exit", "quit"]:
                print("🤖 Goodbye!")
                break
            
            print("🤖 Thinking...")
            
            # Invoke the LangChain process
            response = chain.invoke({"question": user_query})
            
            print(f"\n🤖: {response}")

        except KeyboardInterrupt:
            print("\n🤖 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
