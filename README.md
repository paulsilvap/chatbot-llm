# Chatbot Project

This is a Flask-based chatbot application that uses a PostgreSQL database to store chat history and integrates with Llama for generating responses.

## Features
- Stream responses using Flask
- Store chat history in PostgreSQL
- Integrate with AI models for generating responses

## Setup
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Flask application:
   ```bash
   python chatbot.py
   ```

## Usage
- Send a POST request to `/stream` with a JSON payload containing the user query.
- The server will stream the response back to the client.