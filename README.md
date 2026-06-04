# llama_cpp_python

In this repo, we learn how to use `llama-cpp-python` api to interact with `llama.cpp` backend locally.

You can visite the official [github repo](https://github.com/abetlen/llama-cpp-python) for more information.

## introduction

`llama-cpp-python` is one of the cleanest ways to run LLMs locally because it `binds directly to llama.cpp using ctypes`, 
giving you bare-metal C/C++ performance right inside your Python ecosystem.

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

