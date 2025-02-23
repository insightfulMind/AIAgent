import openai
import os
import speech_recognition as sr
import pyttsx3 
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.playground import Playground, serve_playground_app
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please provide an OpenAI API key")


client = openai.OpenAI(api_key=OPENAI_API_KEY)

engine = pyttsx3.init()

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=8)
            with open("audio.wav", "wb") as f:
                f.write(audio.get_wav_data())

            with open("audio.wav", "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            transcript = response.text   
            print(f"You said:{transcript}")
            return transcript

        except sr.UnknownValueError:
            print("Sorry,I couldn't understand that.")
            return None
        except sr.RequestError:
            print("🔌API unavailable.Please check your internet.")
            return None


def speak_text(text):
    engine.say(text)
    engine.runAndWait()

embedding_model = SentenceTransformerEmbedder(id="sentence-transformers/all-MiniLM-L6-v2")

knowledge_base = JSONKnowledgeBase(
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
    model=OpenAIChat(id="gpt-3.5-turbo", api_key=OPENAI_API_KEY),
    introduction="Hello! Welcome, I am here to assist you. How can I help?",
    save_response_to_file="agent_responses.txt",
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    description="AI Agent for answering customer-related queries about shipment.",
    instructions=[
        "Use a conversational tone like a helpful assistant.",
        "If the knowledge base contains the answer, use it first before generating from AI.",
        "If multiple sources provide similar answers, summarize and return only one.",
        "Do not answer questions that are not related to shipping, logistics, or freight forwarding.",
        "If the user asks an unrelated question, politely refuse to answer."
    ],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=False,
    debug_mode=False,
    markdown=True,
)

if cust_supp_agent.knowledge and os.path.exists("knowledge_base.json"):
    cust_supp_agent.knowledge.load()

app = Playground(agents=[cust_supp_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("audiopaced:app")