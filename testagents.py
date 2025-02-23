import os
from agno.agent import Agent
# from agno.media import Image
from agno.models.groq import Groq
from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please provide a GROQ API key")

embedding_model = SentenceTransformerEmbedder(id="sentence-transformers/all-MiniLM-L6-v2")

knowledge_base=JSONKnowledgeBase(
    path="knowledge_base.json",
    vector_db=LanceDb(
        uri="tmp/lancedb",  # Local storage
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
    ],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    debug_mode=True,
    markdown=True,
)

if cust_supp_agent.knowledge and os.path.exists("knowledge_base.json"):
    cust_supp_agent.knowledge.load()


# cust_supp_agent.print_response("How can I use xhipment for shipping my goods", stream=True)
# cust_supp_agent.print_response("what are the responsibilities of a freight forwarder? ", stream=True)
cust_supp_agent.print_response("what documents are needed for shipping through water carriers? ",stream=True)
