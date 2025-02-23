# AI-Powered Customer Support Agent for shipment  

🚀 Project Overview
This project is an AI-powered **Customer Support Agent**. The agent answers customer-related queries by leveraging a structured **knowledge base (JSON)** and an AI model (**Groq's LLaMA-3.3-70B**). It prioritizes accuracy, customer-friendliness, and compliance with industry regulations.

🛠️ Tech Stack
- **Python** → Core programming language
- **Agno AI Framework** → Manages AI workflows
- **Groq LLaMA-3.3-70B** → AI model for natural language responses
- **LanceDB** → Lightweight vector database for knowledge retrieval
- **Sentence Transformer Embeddings** → For efficient text similarity search



##  How to Set Up and Run
### 1️⃣ Install Dependencies

pip install -r requirements.txt


### 2️⃣ Set Up API Keys
Create a `.env` file and add:

GROQ_API_KEY=your_groq_api_key


### 3️⃣ Run the AI Agent

python testagents.py


## 🤖 Why I Chose This Tech Stack
### 1️⃣ **Groq AI Model**  
- Chosen for **fast inference** & **efficient cost scaling** over OpenAI model as it has limitation on free usuage.

### 2️⃣ **LanceDB for Vector Search**  
- I didnt use postgre here because:
  - PostgreSQL is robust for large-scale applications but adds setup complexity not needed for my task.
  - LanceDB is **lightweight** and works **locally** without external dependencies.
  - I found it Ideal for quick development and deployment.

### 3️⃣ **JSON Knowledge Base Instead of PDF**
- **Why JSON?**
  - JSON enables **structured querying** for **faster, cleaner responses**.
  - PDFs require **preprocessing** and **chunking**, adding potential for errors.
  - JSON allows **easy future updates** to knowledge.

## 🔮 Potential Scope & Improvements
- If this AI scales, migrating to **PostgreSQL + PgVector** would allow **better indexing & performance**.
- Models used currently can be changed to a better model.
- Various other parameters like session management ,storage etc can be added.

### 2️⃣ **Improve Web Search Capabilities**
- Currently Only knowledge base search is used.
- **Future Option:** Integrating **real-time web search** (DuckDuckGo API) for **latest regulations & industry updates**.
- The model isn't much friendly in converstation and sometimes gives a long response.

This AI agent was designed to be **fast, lightweight, and knowledge-driven** for the task, while keeping scalability options open. Future improvements could involve **switching to PostgreSQL for better vector storage**, integrating **real-time search**, and **building an intuitive UI** for better user interaction.

---
✅ **Contributors:** Sakshi Pandey 