# llama_cpp_python

In this repo, we learn how to use `llama-cpp-python` api to interact with `llama.cpp` backend locally.

You can visite the official [github repo](https://github.com/abetlen/llama-cpp-python) for more information.

## Introduction

`llama-cpp-python` is one of the cleanest ways to run LLMs locally because it `binds directly to llama.cpp using ctypes`, 
giving you bare-metal C/C++ performance right inside your Python ecosystem.


## Main features

### Model Initialization

`Llama` class loads the `GGUF model into memory`, with the user provided configuration (e.g. n_gpu_layers, n_ctx, etc.)

Below is a simple example
```python
from llama_cpp import Llama

# Always set n_ctx explicitly so you don't default to small limits.
# Set n_gpu_layers to offload layers to your GPU (-1 offloads all layers), change to 0 if running purely on CPU
Llama(
        model_path=model_path,
        n_threads = 8 # based on your cpu number
        n_ctx=CONTEXT_WINDOW,
        n_gpu_layers=0, 
        embedding=True,  # change to True if you intend to use create_embedding()
        verbose=False  # Turn off to keep stdout clean, turn on for debugging speed
    )
```

### Chat completion

The `chat completion(llm.create_chat_completion)` processes a `structured conversation history` (a list of dictionaries with role and content) 
and returns the model's next response.

Below is a simple example

```python
# when stream=true, the return type is CreateChatCompletionStreamResponse, otherwise is CreateChatCompletionResponse
llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        stream=true
    )
```

> This is your go-to for 90% of applications. Use it for chatbots, interactive assistants, or any task requiring 
> systemic prompting (using a system role to set behavior). It automatically handles the chat templating 
> for the specific model you are using.
> 
> 
### raw text completion

The `raw text completion(llm.create_completion)` accepts a raw string prompt and predicts the text that follows it.

Below is a simple example
```python
response= llm.create_completion(
        prompt=cleaned_prompt,
        max_tokens=128,
        temperature=0.2,  # Lower temperature for more deterministic output
        stop=["\n\n", "###"]  # Stop tokens to prevent the model from rambling
    )
```

> The Use Case is `Text completion, code generation, structured data parsing`, or legacy workflows 
> where you want to manually manage the prompt template wrapper (like Alpaca or Vicuna raw formats).

### embedding generation

The `embedding generation(llm.create_embedding)` converts a piece of text into a vector embedding (a list of floats). 

> The model must be loaded with embedding=True during initialization.
> The use case is to `build Retrieval-Augmented Generation (RAG)`, semantic search, clustering, or building 
> a local vector database pipeline.

### chat completion vs raw_text completion

- Raw text Completion:  generate text from a raw prompt string.
- Chat Completion: generate text from a structured conversation (messages) using a chat template.

Without chat template, user must write everything in the prompt, for example
```python
prompt = """
You are a cybersecurity expert.

User: Explain CVE.
Assistant:
"""
```

With chat template, you can use the below message

```python
messages = [
    {
        "role": "system",
        "content": "You are a cybersecurity expert."
    },
    {
        "role": "user",
        "content": "Explain CVE."
    }
]
```

> The model must support the chat template, otherwise it will not understand the `role`, `content`. And we called the 
> models which support chat template as `chat model`, and tools only works with `chat model`

## Install the packages

Here, we suppose you are using debian 13 as OS.

```shell
sudo apt update
sudo apt install build-essential python3-dev
```

> `build-essential`: Provides gcc, g++, and make, which llama.cpp requires to compile.
> `python3-dev`: Provides Python.h, which is required to link the compiled C++ code to the Python interpreter.
> 
Activate your virtual environment

```shell
source path/to/venv/bin/activate

pip install llama-cpp-python

# if you are using uv to handle your python project, go to your project root folder and run 
uv pip install llama-cpp-python

# if you have nvidia gpu, you need to run
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python

# Add the dependency to project (saves it to pyproject.toml and uv.lock)
CMAKE_ARGS="-DGGML_CUDA=on" uv add llama-cpp-python

```

