# Streamlit RAG Agent with Ollama and Docker

This is a local-first RAG (Retrieval-Augmented Generation) agent that allows you to chat with your PDF documents.

It uses Streamlit for the web UI, LangGraph to create the agent, and local models from Ollama (`qwen3:8b` and `qwen3-embedding:4b`). The entire application is containerized with Docker Compose for a simple, one-command setup.

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **Agent Logic:** LangGraph
* **LLM:** Ollama (serving `qwen3:8b`)
* **Embeddings:** Ollama (serving `qwen3-embedding:4b`)
* **Vector Store:** FAISS (in-memory)
* **Deployment:** Docker Compose

## ⚠️ Prerequisites & Warnings

Before you begin, you must have these tools installed on your system:
* [Git](https://git-scm.com/downloads)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### **Storage Warning**
This project runs models **locally** on your machine. You must have **at least 7-10 GB of free disk space** to download and store the `qwen3` models. This is a one-time download; the models are saved in a Docker volume for future use.

**1. Clone the Repository**
git clone (https://github.com/jhaveri-bhavya/Streamlit-Ollama-PDF-RAG-Agent.git)

```bash
cd Streamlit-Ollama-PDF-RAG-Agent
```

**2. Build and Run the Containers** 
```bash
docker-compose up -d --build
```


**3. Pull the Ollama Models** 
In a new terminal paste the below:  
```bash
# Pull the main chat model (This will take a few minutes)
docker exec -it ollama_server ollama pull qwen3:8b

# Pull the embedding model
docker exec -it ollama_server ollama pull qwen3-embedding:4b
```
**4. Open the App** 
(http://localhost:8501)


🛑 How to Stop
```bash
docker-compose down
```
