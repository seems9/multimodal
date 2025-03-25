# multimodal
Multimodal Receipt Processing with Gemini and Qdrant
Overview

This project processes receipt images, extracts relevant metadata using Google's Gemini MultiModal model, and stores the extracted information in a Qdrant vector database. It also includes a simple Streamlit-based chat application that allows users to query receipt data.

Features

Uses Gemini MultiModal LLM to extract structured data from receipt images.

Stores extracted information in Qdrant, a vector database.

Allows querying of stored receipt data using a Streamlit chat interface.

Retrieves relevant receipts based on user queries using vector similarity search.

Dependencies

Ensure you have the following Python packages installed:

pip install qdrant-client llama-index streamlit pydantic requests matplotlib chromadb

Setup

Install dependencies: Run the above pip install command to install required libraries.

Set up API keys: Ensure you have a valid API_KEY for Google's Gemini model.

Prepare receipt images: Place receipt images in a specified directory.

Run the Streamlit application:

streamlit run multimodal.py

How It Works

Image Processing: The script loads receipt images from a directory.

Data Extraction: Gemini MultiModal extracts structured data including company name, date, address, total amount, currency, and a summary.

Storage in Qdrant: Extracted data is stored as vector embeddings in a Qdrant collection.

Querying Receipts: Users can input questions in the Streamlit interface, and relevant receipt data is retrieved using vector similarity search.

Code Breakdown

get_image_files(): Loads image files from a directory.

ReceiptInfo: Pydantic model defining extracted receipt fields.

pydantic_gemini(): Uses Gemini LLM to process image data.

aprocess_image_file(): Processes a single receipt image.

aprocess_image_files(): Processes multiple receipt images.

get_nodes_from_objs(): Converts extracted data into LlamaIndex TextNode objects.

VectorStoreIndex(): Creates a vector database index in Qdrant.

retriever.retrieve(): Retrieves similar receipts based on user queries.

Streamlit UI components handle user input and display query results.

Future Enhancements

Support for additional receipt formats (PDF, PNG, etc.).

Integration with a database for structured storage.

Enhanced query capabilities with natural language processing.

Improved UI for a better user experience.
