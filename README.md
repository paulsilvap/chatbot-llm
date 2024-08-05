# Chatbot Project

This is a Flask-based chatbot application that uses a PostgreSQL database to store the vector embeddings and the chat history and uses Retrieval Augmented Generation (RAG) to ground LLAMA to generate responses about the "Llama 2: Open Foundation and Fine-Tuned Chat Models" paper.

## Features
- Stream responses using Flask
- Store chat history in PostgreSQL
- Store vectors embeddings in PostgresSQL
- Integrate with AI models for generating responses

## Prerequisites
- Python 3.10 or higher
- PostgreSQL
- pip (Python package installer)

## Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```
2. **Install the required dependencies**: 
   ```bash
    # PostgreSQL requires pgvector for the vector database.
    cd /tmp
    git clone --branch v0.7.3 https://github.com/pgvector/pgvector.git
    cd pgvector
    make
    make install # may need sudo

    # Build llama.cpp with CUDA and OpenBLAS support
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make LLAMA_CURL=1 GGML_OPENBLAS=1 GGML_CUDA=1 CUDA_DOCKER_ARCH=compute_89

    # Build llama-cpp-python with CUDA and OpenBLAS support
    CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

    # Install additional requirements
    pip install -r requirements.txt
   ```
3. **Set up the PostgreSQL server**:
    - Install PostgreSQL if it's not already installed.
    - Create a new user to use on your local version:
        ```bash
        CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';
        ALTER ROLE <user> SUPERUSER;
        ```
    - Use your user and password inside chatbot.py

4. **Run the Flask application**:
   ```bash
   python chatbot.py
    ```

## Evaluation
The responses.txt file contains sample questions and answered retrieved by the llm by using RAG. It also includes the estimated latency of the chatbot on a RTX4070 card.
