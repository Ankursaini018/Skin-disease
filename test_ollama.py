import requests

def test_ollama_api():
    """Test the Ollama API integration"""
    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "gemma3"  # Using llama2 as it's smaller and more commonly available
    
    # Test prompt
    prompt = "Explain athlete's foot in detail and provide treatment options. Please structure your response with:\n\n1. What is athlete's foot? (Brief explanation)\n2. Symptoms and signs\n3. Causes and risk factors\n4. Treatment options and cures (4-5 points)\n5. Prevention tips\n\nPlease keep the response concise but informative, focusing on practical treatment advice."
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        print("Testing Ollama API connection...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Ollama API is working!")
            print("\nResponse preview:")
            print(result["response"][:200] + "...")
            return True
        else:
            print(f"❌ Ollama API error: {response.status_code}")
            print(f"Error message: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama API. Make sure Ollama is running on localhost:11434")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_ollama_api() 