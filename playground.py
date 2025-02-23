import os
from agno.agent import Agent
# from agno.media import Image
from agno.models.groq import Groq
from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.playground import Playground, serve_playground_app

from dotenv import load_dotenv
load_dotenv()

AGNO_API_KEY = os.getenv("AGNO_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please provide a GROQ API key")

embedding_model = SentenceTransformerEmbedder(id="sentence-transformers/all-MiniLM-L6-v2")

knowledge_base=JSONKnowledgeBase(
    path="knowledge_base.json",
    vector_db=LanceDb(
        uri="tmp/lancedb", 
        table_name="knowledge_store",
        search_type=SearchType.hybrid,
        embedder=embedding_model
    ),
    add_history_to_messages=True,
    markdown=True,
)
cust_supp_agent = Agent(
    name="customer_agent",
    model=Groq(id="llama-3.3-70b-versatile",api_key=GROQ_API_KEY),
    introduction="Hello! welcome, I am here to assist you. How can I help?",
    save_response_to_file="agent_responses.txt",
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    description="AI Agent for answering customer related Queries about shipment.",
    instructions=[
        "Use a conversational tone like a helpful assistant.",
        "If the knowledge base contains the answer, use it first before generating from AI.",
        "If multiple sources provide similar answers, summarize and return only one.",
        "Do not answer questions that are not related to shipping, logistics, or freight forwarding.",
        "If the user asks an unrelated question, politely refuse to answer."
        "Keep answer precise and crisp"
    ],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=False,
    debug_mode=False,
    markdown=True,
)

app = Playground(agents=[cust_supp_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)