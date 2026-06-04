import time
from typing import Dict, Generator, List, Any
from llama_cpp import Llama
from basic.llama_cpp_api import initialize_llm
from conf.constant import MODEL_PATH


def run_chat_completion_with_metrics(llm: Llama, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Executes a chat completion, streams tokens to stdout, and calculates performance metrics.

    Returns:
        A dictionary containing the full text response and performance benchmarks.
    """

    # 1. Record the exact timestamp before passing the prompt to the model
    start_time = time.perf_counter()

    # Request the stream
    stream_response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        stream=True
    )

    first_token_time = None
    token_count = 0
    full_text = []

    print("AI Response: ", end="", flush=True)

    # 2. Iterate through the stream generator
    for chunk in stream_response:
        delta = chunk["choices"][0]["delta"]

        if "content" in delta:
            # Capture the absolute time the very first token escapes the generator
            if first_token_time is None:
                first_token_time = time.perf_counter()

            token_text = delta["content"]
            print(token_text, end="", flush=True)

            full_text.append(token_text)
            token_count += 1

    # 3. Record the end timestamp when the generator terminates
    end_time = time.perf_counter()
    print("\n" + "=" * 40)

    # --- Metrics Logic ---
    # Time to First Token (TTFT): From prompt start to first token output
    ttft_seconds = first_token_time - start_time if first_token_time else 0.0

    # Generation Time: Time spent ONLY generating subsequent tokens
    generation_time_seconds = end_time - first_token_time if first_token_time else 0.0

    # Tokens per Second (TPS): Total output tokens divided by generation duration
    # Best Practice: Exclude TTFT from this calculation to avoid corrupting generation speeds
    tokens_per_second = (token_count / generation_time_seconds) if generation_time_seconds > 0 else 0.0

    return {
        "text": "".join(full_text),
        "total_tokens": token_count,
        "ttft_ms": ttft_seconds * 1000,  # Typically measured in milliseconds
        "tokens_per_second": tokens_per_second
    }


if __name__ == "__main__":
    print("Initializing LLM...")
    try:
        llm: Llama = initialize_llm(MODEL_PATH)
        print("Model loaded successfully.\n" + "=" * 40)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    conversation = [
        {"role": "system", "content": "You are a helpful software architecture assistant."},
        {"role": "user", "content": "Explain the architectural difference between REST and gRPC in two sentences."}
    ]

    # Execute and capture benchmarks
    metrics = run_chat_completion_with_metrics(llm, conversation)

    # Display the profiled data cleanly
    print(f"📊 PERFORMANCE METRICS:")
    print(f"  • Time to First Token (TTFT): {metrics['ttft_ms']:.2f} ms")
    print(f"  • Generation Speed:          {metrics['tokens_per_second']:.2f} tokens/sec")
    print(f"  • Total Tokens Emitted:      {metrics['total_tokens']} tokens")