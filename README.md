# RAG System for "Attention Is All You Need" Paper

This repository contains a Retrieval Augmented Generation (RAG) system designed to answer questions based on the "Attention Is All You Need" research paper. It leverages LangChain for orchestration, Google's Gemini-1.5 Pro for generation and embeddings, and MongoDB Atlas Vector Search for efficient document retrieval.

---

## Features

* **Document Ingestion:** Automatically loads the `attention.pdf` document, splits it into chunks, generates embeddings using Google's `embedding-001` model, and stores them in your MongoDB Atlas vector store.
* **Vector Search:** Utilizes MongoDB Atlas Vector Search for semantic similarity search to retrieve the most relevant document chunks based on a user's query.
* **Large Language Model (LLM):** Integrates with Google's **Gemini-1.5 Pro** model for generating coherent and contextualized answers.
* **LangChain Integration:** Built using LangChain Expression Language (LCEL) for a modular and readable RAG pipeline.
* **Prompt Management:** Fetches the RAG prompt template directly from LangChain Hub (`rlm/rag-prompt`) for best practices.
* **Secure Credential Handling:** Uses environment variables loaded via `python-dotenv` to keep your API keys and connection strings secure and out of your codebase.

---

## Prerequisites

Before you can run this project, make sure you have the following set up:

* **Conda:** For managing your Python environment.
* **Python 3.9+:** (Recommended for Conda environment creation).
* **MongoDB Atlas Account:**
    * A MongoDB Atlas cluster.
    * A database (e.g., `RegPractise`) and a collection (e.g., `attention`) within that cluster.
    * **A Vector Search Index:** Create a vector search index named `vector_index` (or whatever you set `vector_index_name` to in `rag_pipeline.py`) on your `attention` collection. This index must be configured to index the `embedding` field.
        * **MongoDB Atlas Vector Search Index Definition JSON:**

        ```json
        {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 768,
                    "similarity": "cosine"
                    }
                    ]
                    }

        ```
* **Google Cloud Project & API Key:**
    * You need a Google Cloud Project.
    * Enable the **"Generative Language API"** within your project.
    * Generate an API Key from the Google Cloud Console.
* **LangSmith Account (Optional but Recommended):**
    * For advanced tracing, debugging, and evaluation of your LangChain application.
    * Sign up at [smith.langchain.com](https://smith.langchain.com/).

---

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git) # Replace with your actual repo URL
    cd your-repo-name # Navigate into your project directory
    ```

2.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n rag_env python=3.9 -y
    conda activate rag_env
    ```

3.  **Place Your PDF Document:**
    * Make sure your `attention.pdf` file is placed directly in the root directory of your project (where `rag_pipeline.py` is located).

4.  **Configure Environment Variables:**
    * Create a new file named `.env` in the root directory of your project.
    * Add your credentials and configurations to this `.env` file. **Replace the placeholder values with your actual keys and URI.**
        ```
        MONGO_URI="mongodb+srv://shubham_agrawal:<YOUR_MONGO_PASSWORD>@regpractise.wqnkfz3.mongodb.net/?retryWrites=true&w=majority&appName=RegPractise"
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        LANGCHAIN_API_KEY="YOUR_LANGCHAIN_SMITH_API_KEY" # Optional, only if using LangSmith
        LANGCHAIN_TRACING_V2="true" # Optional, only if using LangSmith
        LANGCHAIN_PROJECT="RAG-Attention-Paper" # Optional, set your project name for LangSmith
        ```
    * **Crucial:** Add `.env` to your `.gitignore` file to prevent accidentally committing your credentials.

5.  **Install Dependencies:**
    * With your `rag_env` active, install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
        *(If you haven't generated `requirements.txt` yet, run `pip install langchain pymongo langchain-google-genai pypdf langchain-mongodb python-dotenv langchainhub` and then `pip freeze > requirements.txt`)*.

---

## Usage

1.  **Run the RAG Pipeline Script:**
    * Ensure your `rag_env` Conda environment is active.
    * Execute the `rag_pipeline.py` script:
        ```bash
        python rag_pipeline.py
        ```
    * **First Run:** On the first run, the script will:
        * Connect to MongoDB Atlas.
        * **Clear existing documents** in the specified collection (`collection.delete_many({})`).
        * Ingest the `attention.pdf` by splitting pages, generating embeddings, and storing them in your vector store. This step might take a few minutes depending on the PDF size and network speed.
    * **Subsequent Runs:** If you run the script again, it will repeat the ingestion process (clearing and re-adding documents).
    * After ingestion, the script will:
        * Initialize the RAG pipeline.
        * Execute the example queries defined in the `if __name__ == "__main__":` block.
        * Print the generated answers to your terminal.

---

## Important Notes

* **Credential Security:** Always keep your `.env` file out of version control (`.gitignore` is set up for this). Never hardcode sensitive information directly in your scripts.
* **MongoDB Atlas Vector Index:** Ensure your vector search index is correctly configured and built in MongoDB Atlas *before* running this script, as the ingestion process relies on it.
* **PDF Name:** The script expects the PDF file to be named `attention.pdf` and located in the same directory as `rag_pipeline.py`.
* **Ingestion Behavior:** The `collection.delete_many({})` line ensures a fresh ingestion every time the script runs. For a production system, you might want to remove or comment this line after the initial ingestion, or implement a more sophisticated check to prevent re-ingesting data unnecessarily.

---

## Potential Enhancements

* **Advanced Evaluation:** Integrate more sophisticated evaluation metrics for both retrieval (e.g., Recall, MRR) and generation (e.g., Faithfulness, Answer Relevance) using LangSmith's built-in evaluators.
* **Batch Processing:** Implement batch ingestion for larger datasets.
* **API Exposure:** Wrap the RAG pipeline in a web API (e.g., with Flask or FastAPI) to create a user-facing application.