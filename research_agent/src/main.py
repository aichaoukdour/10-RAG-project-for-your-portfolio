import sys
import os

# Add the parent directory to sys.path so we can import 'src' as a package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import ShortResearchAgent

def main():
    agent = ShortResearchAgent()
    
    print("\n" + "="*50)
    print("      🧪 AI RESEARCH AGENT - AUTOMATED RESEARCH")
    print("="*50)
    
    default_query = "What causes urban heat islands and how can cities reduce them?"
    
    print(f"\nEnter your research query (or press Enter for default):")
    query = input("> ").strip()
    
    if not query:
        query = default_query
        
    print(f"\n🚀 Running automated research for: '{query}'")
    print("Thinking... This may take a few seconds.\n")
    
    try:
        result = agent.run(query)
        
        print("\n" + "-"*50)
        print("📝 EXTRACTIVE SUMMARY")
        print("-"*50)
        print(result["summary"])
        print("-"*50)
        
        print("\n🔍 TOP RELEVANT PASSAGES")
        for i, p in enumerate(result["passages"], 1):
            print(f"\n[{i}] Score: {p['score']:.3f} | Source: {p['url']}")
            print(f"    {p['passage'][:300]}...")
            
        print("\n" + "="*50)
        print(f"✅ Research completed in {result['time']:.1f}s")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Research cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
