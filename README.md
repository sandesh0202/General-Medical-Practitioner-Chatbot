# Medical Chat Interface Application

## Introduction

This application is designed to process medical reports in PDF format and provide a chat interface where users can ask questions related to their medical reports. The application uses advanced natural language processing (NLP) techniques to analyze the reports and respond to user queries.
<img width="879" alt="Screenshot 2024-05-25 224601" src="https://github.com/sandesh0202/General-Medical-Practitioner-Chatbot/assets/74035326/d0f92f55-31c9-4ffa-bb1b-905f95109b0a">
<img width="892" alt="Screenshot 2024-05-25 224529" src="https://github.com/sandesh0202/General-Medical-Practitioner-Chatbot/assets/74035326/81e7dab9-1024-4d1e-a0b9-a02e66084990">

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Application Workflow](#application-workflow)
5. [Explanation of Key Components](#explanation-of-key-components)
6. [How to Use the Application](#how-to-use-the-application)
7. [Running the Application](#running-the-application)

## Prerequisites

Before running this application, make sure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the Repository**:

    ```sh
    git clone https://github.com/sandesh0202/General-Medical-Practitioner-Chatbot.git
    ```

2. **Install the Required Python Packages**:

    ```sh
    pip install -r requirements.txt
    ```

## Project Structure

Here's an overview of the project structure:


Certainly! Below is the markdown code for the README file as requested.

markdown
Copy code
# Medical Chat Interface Application

## Introduction

This application is designed to process medical reports in PDF format and provide a chat interface where users can ask questions related to their medical reports. The application uses advanced natural language processing (NLP) techniques to analyze the reports and respond to user queries.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Application Workflow](#application-workflow)
5. [Explanation of Key Components](#explanation-of-key-components)
6. [How to Use the Application](#how-to-use-the-application)
7. [Running the Application](#running-the-application)

## Prerequisites

Before running this application, make sure you have the following installed:

- Python 3.12
- pip (Python package installer)

## Installation

1. **Clone the Repository**:

    ```sh
    git clone https://github.com/yourusername/medical-chat-interface.git
    ```

2. **Install the Required Python Packages**:

    ```sh
    pip install -r requirements.txt
    ```

## Project Structure

Here's an overview of the project structure:

General-Practitioner/
│
├── data/
│ └── output.md # Markdown file generated from PDF
│
├── uploads/ # Directory to store uploaded PDF files
│
├── app.py # Main application file
├── index.html # Main landing page HTML
├── chat.html # Chat interface HTML
├── requirements.txt # List of required Python packages
└── README.md # This README file


## Application Workflow

1. **File Upload**: Users upload their medical report PDFs via the web interface.
2. **PDF Processing**: The uploaded PDF is converted to markdown text for easier processing.
3. **Text Chunking**: The markdown text is split into manageable chunks for efficient processing.
4. **Vector Store Creation**: These chunks are converted into vector representations for semantic search.
5. **Chat Interface**: Users ask questions, and the system retrieves relevant information from the processed text and generates responses.

## Explanation of Key Components

### 1. `get_documents()`

This function handles the PDF parsing using the `LlamaParse` library. It either loads previously parsed documents from a pickle file or parses a new PDF and saves the result for future use.

### 2. `get_markdown_text()`

This function converts the parsed documents into markdown text and splits them into chunks using `RecursiveCharacterTextSplitter`. This step is crucial for managing large documents efficiently.

### 3. `get_vectorstore()`

This function creates a vector store using the `Qdrant` library. Vector stores allow for efficient semantic search, enabling the application to find relevant document sections based on user queries.

### 4. `Chain()`

This function sets up a retrieval-augmented generation (RAG) chain. It combines a retriever, which searches the vector store, with a language model (LLM) to generate responses based on the retrieved context.

### 5. Flask Routes

- **`/`**: Renders the main landing page (`index.html`).
- **`/process`**: Handles PDF upload, saves the file, and redirects to the chat interface.
- **`/chat`**: Handles user queries, invokes the LLM chain to generate responses, and renders the chat interface (`chat.html`).

## How to Use the Application

1. **Upload a PDF**: Go to the main page and upload your medical report PDF.
2. **Ask Questions**: Once the PDF is processed, you'll be redirected to the chat interface. Here, you can select the model and ask questions about your medical report.
3. **Receive Answers**: The application will provide detailed answers based on the contents of your medical report.

## Running the Application

1. **Start the Flask Server**:

    ```sh
    python app.py
    ```

2. **Access the Application**:

    Open your web browser and go to `http://localhost:8080`.

3. **Upload a PDF and Start Chatting**:

    - Upload your PDF.
    - Ask questions in the chat interface.
    - Receive detailed responses based on your medical report.

## Detailed Explanation of Concepts

### Natural Language Processing (NLP)

NLP is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. In this application, NLP techniques are used to understand and process the medical report's content.

### Vector Stores and Semantic Search

Vector stores convert text into numerical representations (vectors) that capture the semantic meaning of the text. This allows the application to perform semantic search, finding the most relevant sections of the document based on the user's query.

### Retrieval-Augmented Generation (RAG)

RAG combines retrieval and generation in NLP. First, it retrieves relevant document sections using a retriever (vector store). Then, it generates a response using a language model, ensuring the answer is both relevant and contextually accurate.

### Language Models (LLMs)

LLMs like GPT, Llama3, and Mixtral are used to generate human-like text based on the input they receive. In this application, different LLMs can be selected to generate responses based on the retrieved context.

---

By following this guide, you should be able to understand the core components and workflow of the Medical Chat Interface Application. If you encounter any issues or have further questions, please refer to the documentation of the libraries used or seek help from the community.
