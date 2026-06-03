import os
from typing import Any, Dict, Generator, List
from llama_cpp import Llama

# Best Practice: Use constants or environment variables for configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./models/llama-3-8b-instruct.Q4_K_M.gguf")
CONTEXT_WINDOW = 4096


def initialize_llm(model_path: str) -> Llama:
    """Initializes and returns the Llama backend.

    Best Practice:
    - Always set n_ctx explicitly so you don't default to small limits.
    - Set n_gpu_layers to offload layers to your GPU (-1 offloads all layers).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please download the GGUF file first."
        )

    return Llama(
        model_path=model_path,
        n_ctx=CONTEXT_WINDOW,
        n_gpu_layers=-1,  # Change to 0 if running purely on CPU
        embedding=True,  # Required if you intend to use create_embedding()
        verbose=False  # Turn off to keep stdout clean, turn on for debugging speed
    )


# =====================================================================
# Function 1: Chat Completion (Standard & Streaming)
# =====================================================================
def run_chat_completion(
        llm: Llama,
        messages: List[Dict[str, str]],
        stream: bool = False
) -> Any:
    """Handles conversational interactions using structural chat formats.

    Use case: Conversational UI, agents, or role-based tasks.
    """
    # create_chat_completion mirrors the OpenAI format
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        stream=stream
    )
    return response


# =====================================================================
# Function 2: Raw Text Completion
# =====================================================================
def run_raw_completion(llm: Llama, prompt: str) -> str:
    """Predicts next tokens from a raw string prompt.

    Use case: Text transformation, code completion, or raw text generation.
    """
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=128,
        temperature=0.2,  # Lower temperature for more deterministic output
        stop=["\n", "###"]  # Stop tokens to prevent the model from rambling
    )
    return response["choices"][0]["text"].strip()


# =====================================================================
# Function 3: Text Embeddings
# =====================================================================
def get_text_embedding(llm: Llama, text: str) -> List[float]:
    """Generates numerical vector embeddings for a given text.

    Use case: Semantic search or feeding into a vector database.
    """
    embedding_data = llm.create_embedding(input=[text])
    return embedding_data["data"][0]["embedding"]


# =====================================================================
# Execution Execution / Workflow Demonstration
# =====================================================================
if __name__ == "__main__":
    print("Initializing LLM...")
    try:
        llm = initialize_llm(MODEL_PATH)
        print("Model loaded successfully.\n" + "=" * 40)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # 1. Chat Completion Example (Streaming)
    print("\n--- Testing Chat Completion (Streaming) ---")
    chat_history = [
        {"role": "system", "content": "You are a witty, concise senior Python developer."},
        {"role": "user", "content": "Why do Pythonistas love list comprehensions?"}
    ]

    # We use stream=True for smooth UI updates
    stream_response = run_chat_completion(llm, chat_history, stream=True)

    print("AI Response: ", end="", flush=True)
    for chunk in stream_response:
        # Extract the text token from the delta dictionary safely
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)
    print("\n")

    # 2. Raw Completion Example
    print("\n--- Testing Raw Text Completion ---")
    raw_prompt = "def calculate_factorial(n):\n    if n == 0:\n        return 1"
    completed_code = run_raw_completion(llm, raw_prompt)
    print(f"Prompt:\n{raw_prompt}\nCompleted Code:\n{completed_code}\n")

    # 3. Embedding Example
    print("\n--- Testing Text Embedding ---")
    sample_text = "Python is an interpreted, high-level language."
    vector = get_text_embedding(llm, sample_text)
    print(f"Generated Vector length: {len(vector)}")
    print(f"First 5 dimensions: {vector[:5]}")