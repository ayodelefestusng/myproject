# Django and settings
from django.conf import settings
from django.templatetags.static import static

# Standard Library Imports
import os
import sys
import uuid
import random
import sqlite3
import datetime
import json
import glob
from pprint import pprint
from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated

# Third-Party Imports
import psycopg2
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing_extensions import TypedDict

# Google Generative AI
from google import genai
from google.genai import types

# LangChain & LangGraph Imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, AnyMessage, RemoveMessage, trim_messages
)
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.config import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.document_loaders import PyPDFLoader, WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq

from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.utilities import SQLDatabase
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import NodeInterrupt

# Tavily Client
from tavily import TavilyClient

# Project Models
from .models import Insight, SessionData, Sentiment, Checkpoint
import os
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Retrieve variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH", "default.pdf")

# Example usage
print(f"Google API Key: {GOOGLE_API_KEY}")
print(f"PDF Path: {PDF_PATH}")

GOOGLE_API_KEY
gemni = ChatGoogleGenerativeAI(
    # model="gemini-2.5-flash-preview-04-17",
    model="gemini-2.0-flash-exp-image-generation",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# messages = [("system", "You are a helpful assistant that translates English to French. Translate the user sentence.",),
#        ("human", "I love programming."),]
# ai_msg = gemni.invoke(messages)
# print (ai_msg.content)



llm=gemni
llm=llm
model=llm



embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = InMemoryVectorStore(embeddings)
tavily_client = TavilyClient()

DATABASE_URI = os.getenv("DATABASE_URI", "postgresql://postgres:postgres-user-password@75.119.151.28:5432/postgres")
db = SQLDatabase.from_uri(DATABASE_URI)

class Answer(BaseModel):
    """Response schema for user questions"""
    response: str = Field(description="The response to the user's question")
    sources: List[str] = Field(description="List of sources used including PDF content or web search results")
    channels: List[str] = Field(description="Specific channels referenced like POS, ATM, or Web") 
    sentiment: int = Field(description="Sentiment analysis score (-2 to +2)")

class Summary(BaseModel):
    """Conversation summary schema"""
    summary: str = Field(description="Summary of the entire conversation")
    main_channels: List[str] = Field(description="Channels with unresolved issues")
    overall_sentiment: int = Field(description="Sentiment analysis of entire conversation")
    all_sources: List[str] = Field(description="All sources referenced in conversation")

class State(MessagesState):
    """State management for conversation flow"""
    question: str
    pdf_content: List[str]
    web_content: List[str]
    answer: Answer
    summary: Summary
    query_answer: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

def get_time_based_greeting():
    """Return appropriate greeting based on current time"""
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 22:
        return "Good evening"
    return "Good night"



def process_message(message: str,username,session):
    """Main function to process user messages"""
    print ("Start")
    # Find the latest uploaded file in the chat_attached (media) folder

    
    ayula=message
    user_input = ayula
        

    def load_and_search_pdf(state: State):
        """Load and search PDF document"""
        print("--Loading PDF--")
        # file_path = r"C:\Users\User\Desktop\25052025\ATB Bank Nigeria Groq v2.pdf"
        # file_path = os.path.join(os.path.dirname(__file__), 'pdfs', 'sample.pdf')
        # file_path = os.path.join(settings.MEDIA_ROOT, 'pdfs', 'sample.pdf')
        
        file_path = os.path.join(settings.MEDIA_ROOT, 'pdfs', 'ATB Bank Nigeria Groq v2.pdf')

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = text_splitter.split_documents(documents)
            vector_store.add_documents(documents=all_splits)
            # retrieved_docs = vector_store.similarity_search(state["messages"][-1].content)
            retrieved_docs = vector_store.similarity_search(user_input)
            print ("The Print the Message:Joor",state["messages"][-1].content)
            return {"pdf_content": retrieved_docs}
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return {"pdf_content": []}

    def search_web(state: State):
        """Perform web search"""
        try:
            tavily_search = TavilySearchResults(max_results=3)
            # search_docs = tavily_search.invoke(state["messages"][-1].content)
            search_docs = tavily_search.invoke(user_input)
            
            if any(error in str(search_docs) for error in ["ConnectionError", "HTTPSConnectionPool"]):
                return {"web_content": []}
                
            formatted_docs = "\n\n---\n\n".join(
                f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
                for doc in search_docs
            )
            return {"web_content": [formatted_docs]}
        except Exception as e:
            print(f"Web search error: {e}")
            return {"web_content": []}

    def generate_response(state: State):
        """Generate structured response using context"""
        print("--Generating Answer--")
        
        # Prepare context
        pdf_text = "\n".join(doc.page_content for doc in state["pdf_content"])
        web_text = "\n".join(state["web_content"])
        # print ("Anyagde:",state)
        query_answer=state["query_answer"]
        print ("Ayo",query_answer)
        context = f"PDF Content:\n{pdf_text}\n\nWeb Content:\n{web_text}\n\nQuery Answer:\n{query_answer}"
        
        # Prepare prompt template
        greeting = get_time_based_greeting()
        prompt1 = f"""
You are Damilola, the AI assistant for ATB Bank. {greeting}! How may I help you today?

**Instructions:**
1. Provide professional, accurate responses
2. Use the following context:
{context}
3. Structure your response as:
- Answer: Clear response to the question
- Sources: References used
- Channels: Relevant banking channels
- Sentiment: User sentiment score (-2 to +2)

**Question:** {state["messages"][-1].content}
"""
        
        
        prompt = f"""
    "You are Damilola, the AI-powered virtual assistant and Data Analyst for ATB Bank, dedicated to providing professional, accurate, and courteous customer support. Use the provided context to answer the user's question in a structured JSON format. Ensure responses are polite and use the language of the user.",
    Question: {ayula}
    Context: {context}
    "OutputFormat": 
        "answer": "str",
        "sources": "List[str]",
        "channel": "List[str]",
        "sentiment": "int"
    
    "Definitions": 
        "answer": "A clear and concise response to the user's question.",
        "sources": "List of sources used to generate the response, including PDF content, web search or Query  results .",
        "channel": "The specific channels referenced in the conversation, such as POS, ATM, or Web.",
        "sentiment": "A rating of the user's conversation experience, ranging from -2 (very bad) to +2 (very good)."
    instructions:
    "RoleAndBehavior": 
        "Always introduce yourself politely based on the current time ::{greeting}(e.g., 'Good morning and welcome to ATB Bank. I’m Damilola. How can I assist you today?').",
        "Emojis": "Use emojis to express emotions appropriate for the tone of the user's input."
        "InformationHandling": 
        
            "PDFQueries": "Provide precise answers using the document.",
            "ExternalQueries": "Utilize an internet search tool for up-to-date information.",
            "DatabaseQueries": "Utilize an Sql Query search tool for database or analtyics related information.",
            "UnresolvedIssues": "If an issue can't be resolved, escalate it and inform the user courteously."
        ,
        "ComplaintHandling": 
            "Acknowledgment": "Express empathy when responding to complaints.",
            "Acknowledgment": "Use the tools Web and Pdf as guide on resolving customer compplain or inquiry. if customer is still not satified or the required information to resoluve the user issue is not in the tool, inform the customer that the issue would be escaclated to the support .",
            "ResolutionUpdate": "Communicate the action taken to resolve the issue."
        ,
        "CustomerEngagement": 
            "PositiveFeedback": "Thank customers for their kind words.",
            "Apology": "Sincerely apologize for any dissatisfaction."
        ,
        "Closing": "End interactions politely, asking if the user needs further assistance.",
        "Emojis": "Use emojis to express emotions appropriate for the tone of the user's input."
        "Language": "Use the user's preferred language for responses.", 
        "Committment": "Response must indicate thaat the assistant is a member of the company. eg we offer loan",
        "Politeness": "Maintain a polite and professional tone throughout the conversation.",
"""
          

        
        
        
        try:
            sys_msg = SystemMessage(content=prompt)
            model_with_structure = model.with_structured_output(Answer)
            response = model_with_structure.invoke([sys_msg] + state["messages"])
            return {
                "answer": response,
                "sources": response.sources,
                "channels": response.channels,
                "sentiment": response.sentiment
            }
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "I apologize, I'm having trouble answering that question.",
                "sources": [],
                "channels": [],
                "sentiment": 0
            }

    def summarize_conversation(state: State):
        """Generate conversation summary"""
        prompt1 = f"""
Summarize this conversation between Damilola (ATB Bank assistant) and a customer:

**Conversation:**
{state["messages"]}

**Summary Requirements:**
1. Key points discussed
2. Unresolved issues (if any)
3. Overall sentiment (-2 to +2)
4. All sources referenced
"""

        prompt = f"""  
        The chatbot is designed to analyze entire conversations:{state["messages"]}  between a virtual assistant and a user, generating a structured JSON summary that captures unresolved concerns, sentiment, and sources referenced in the discussion.

### **Analysis Approach:**  
1. **Identify Key Topics**: Extract areas of interest or complaints mentioned by the user.  
2. **Track Resolution Status**:  
   - Only unresolved issues in the summary should be included in `mchannel`.  
   - Any issue the assistant has fully resolved should NOT  be noted in `mchannel`.  
3. **Sentiment Evaluation**:  
   - Perform sentiment analysis on the entire conversation:{state["messages"]} .  
   - Give higher priority to the last messages sent by the user for an accurate sentiment assessment.  
4. **Source Attribution**: List all references, including external documents, PDFs,  web search, Query  results.

### **Expected Output Format:**  
```json
''
  "summary": "A concise overview of the entire conversation.",
  "mchannel": ["List of only unresolved concerns, complaints, or interests"], the ouput eg [ATM, POS, Web]"
  "msentiment: -2 to +2,  // Sentiment score (-2: very negative, +2: very positive),"
  "msources": ["List of sources referenced throughout the conversation"]"
```

### **Definitions:**  
- **summary**: A clear and concise summary of the conversation.  
- **mchannel**: Specific unresolved complaints or interests (e.g., POS, ATM, Web).  
- **msentiment**: Sentiment rating (-2 to +2), giving higher priority to the user's latest messages.  
- **msources**: List of sources used, including PDFs or web search results.

---
"""


        try:
            model_with_structure = model.with_structured_output(Summary)
            # response = model_with_structure.invoke([sys_msg] +(statement))
            response = model_with_structure.invoke([SystemMessage(content=prompt)]+state["messages"])
                
            # response = model_with_structure.invoke([SystemMessage(content=prompt)])
            return {
                "summary": response.summary,
                "main_channels": response.main_channels,
                "overall_sentiment": response.overall_sentiment,
                "all_sources": response.all_sources
            }
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return {
                "summary": "Unable to generate summary",
                "main_channels": [],
                "overall_sentiment": 0,
                "all_sources": []
            }
            
            
    
    def write_query(state: State):
        """Generate SQL query to fetch information."""
        # input= state["messages"][-1].content
        input= user_input
        user_prompt = "Question: {input}"
        
     

        system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""
        

        query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                # "input": state["messages"][-1].content,
                "input": user_input,
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        resultT = execute_query_tool.invoke(result["query"])
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            # f'Question: {state["messages"][-1].content}\n'
            f'Question: {user_input}\n'
            f'SQL Query: {result}\n'
            f'SQL Result: {resultT}'
        )
        responseY = llm.invoke(prompt)
        print(f"Answer: {responseY.content}")
        return {"query_answer": responseY.content}
        # return {"query": result["query"], "result": resultT, "query_answer": responseY.content}
        
    def prepare_final_output(state: State) -> dict:
        """Prepare the final output response with all conversation data.
        
        Args:
            state: The current conversation state containing messages, answers, and summaries
            
        Returns:
            dict: Contains the formatted response message and all metadata
        """
        try:
            # Extract the latest question and answer
            # current_question = state["messages"][-1].content
            current_question = user_input
            initial_question = state["messages"][0].content
            answer_data = state["answer"]
            
            # Create structured Answer object
            print ("AGOBA Answer",Answer )
            responseT = Answer(
                response=answer_data.response,  # Changed from 'yeild' to 'response'
                sources=state.get("sources", []),
                channels=state.get("channels", []),  # Changed from 'channel' to 'channels'
                sentiment=state.get("sentiment", 0)
            )
            print ("AGOBA",responseT )
            # print ("IJEBU",responseT )
            # Prepare summary data
            summary_data = {
                "question": initial_question,
                "answer": responseT.response,
                "sentiment": responseT.sentiment,
                "sources": responseT.sources,
                "channels": responseT.channels,
                "conversation_summary": state.get("summary", ""),
                "unresolved_channels": state.get("mchannel", []),  # Unresolved issues
                "overall_sentiment": state.get("msentiment", 0),
                "all_sources": state.get("msources", [])
            }
            # print ("UJEBU SUMMARY",summary_data )
 # For storing list of strings
    
            # For database storage (commented out as in original)
            # from .models import Insight
            # Insight.objects.create(
            #         session_id=1,  # Consider using a dynamic session ID
            #         sentimentAnswer=response.sentiment,
            #         answer=response.response,  # Fixed typo from 'respose' to 'response'
            #         source=json.dumps(response.sources),
            #         ticket=json.dumps(response.channels),
            #         summary=state.get("summary", ""),
            #         sentiment=summary_data["overall_sentiment"],
            #         # mchannel=json.dumps(summary_data["unresolved_channels"]),
            #         # msources=json.dumps(summary_data["all_sources"]),
            #         created_at=datetime.datetime.now(),
            #         question=initial_question,
            #         username=username
            #     )
            
            
            
        
            """Create a new insight if session_id doesn't exist, otherwise update the existing entry."""
            session_id= session
            insight, created = Insight.objects.get_or_create(session_id=session_id)

                # Update only if the entry already exists
            if not created:
                    insight.sentimentAnswer = responseT.sentiment
                    insight.answer = responseT.response  # Fixed typo from 'respose' to 'response'
                    insight.source = json.dumps(responseT.sources)
                    insight.ticket = json.dumps(responseT.channels)
                    insight.summary = state.get("summary", "")
                    insight.sentiment = summary_data["overall_sentiment"]
                    # insight.mchannel = json.dumps(summary_data["unresolved_channels"])
                    # insight.msources = json.dumps(summary_data["all_sources"])
                    insight.updated_at = datetime.datetime.now()  # Update timestamp
                    insight.question = initial_question
                    insight.username = username
                    insight.save()
                    # return f"Insight updated successfully for session {session_id}"

            else:
                    # Creating a new insight entry
                    insight.sentimentAnswer = responseT.sentiment
                    insight.answer = responseT.response
                    insight.source = json.dumps(responseT.sources)
                    insight.ticket = json.dumps(responseT.channels)
                    insight.summary = state.get("summary", "")
                    insight.sentiment = summary_data["overall_sentiment"]
                    # insight.mchannel = json.dumps(summary_data["unresolved_channels"])
                    # insight.msources = json.dumps(summary_data["all_sources"])
                    insight.created_at = datetime.datetime.now()
                    insight.question = initial_question
                    insight.username = username
                    insight.save()

            print("Heloa ",responseT.response)
            return {
                "messages": responseT.response,
                "metadata": summary_data  # Include all structured data
            }

        except KeyError as e:
            print(f"Missing expected state key: {e}")
            return {
                "messages": "I encountered an error processing your request.",
                "metadata": {"error": str(e)}
            }
        except Exception as e:
            print(f"Unexpected error in prepare_final_output: {e}")
            return {
                "messages": "I'm having trouble completing this request.",
                "metadata": {"error": "Internal processing error"}
            }




    # Setup workflow
    DB_URI = "postgresql://postgres:postgres-user-password@75.119.151.28:5432/postgres?connect_timeout=10"
    
    with PostgresSaver.from_conn_string(DB_URI) as memory:
        memory.setup()
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("search_web", search_web)
        workflow.add_node("search_document", load_and_search_pdf)
        workflow.add_node("write_query", write_query)
        workflow.add_node("generate_answer", generate_response)
        workflow.add_node("summarize", summarize_conversation)
        workflow.add_node("final", prepare_final_output)

        # Define workflow
        workflow.add_edge(START, "search_document")
        workflow.add_edge(START, "search_web")
        workflow.add_edge(START, "write_query")
        workflow.add_edge("search_document", "generate_answer")
        workflow.add_edge("search_web", "generate_answer")
        workflow.add_edge("write_query", "generate_answer")
        workflow.add_edge("generate_answer", "summarize")
        workflow.add_edge("summarize", "final")
        workflow.add_edge("final", END)

        # Execute workflow
        graph = workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "216520"}}
        input_message = HumanMessage(content=message)
        output = graph.invoke({"messages": [input_message]}, config)
        
        return {
            "messages": output["messages"][-1].content,
            "metadata": output.get("metadata", {})
        }

def process_message2(message: str,username,session,file_path):
    """Main function to process user messages"""
    print ("Start")
    # Find the latest uploaded file in the chat_attached (media) folder

    attachment = file_path
    ayula=message
    GOOGLE_API_KEY

    client = genai.Client(api_key=GOOGLE_API_KEY)
    safety_settings = [ types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH", ),]

    system_instruction="""
        You are an expert software developer and a helpful coding assistant.
        You are able to generate high-quality code in any programming language.
        """
    config=types.GenerateContentConfig(temperature=0.4,top_p=0.95, top_k=20,candidate_count=1, seed=5,
                                        max_output_tokens=100, stop_sequences=["STOP!"],presence_penalty=0.0,
                                        frequency_penalty=0.0,safety_settings=safety_settings,system_instruction=system_instruction,)



    # img_path = r"C:\Users\User\Downloads\CamScanner 06-04-2025 14.35.jpg"
        # img_path = r"C:\Users\User\Desktop\25052025\ATB Bank Nigeria Groq v2.pdf"

    image = Image.open(attachment)
    image.thumbnail([512,512])
    prompt = "Write out the content of the picture."
    response = client.models.generate_content(model="gemini-2.0-flash", contents=[image, prompt],config=config)
    file_contents= response.text
    print (response.candidates)

    # print (response.candidates.content.parts)
    user_input = f"User Query:\n{ayula}\n\nAttached File Content:\n{file_contents}"

    # Ensure file_contents is accessible in generate_response
    local_file_contents = file_contents
        

    def load_and_search_pdf(state: State):
        """Load and search PDF document"""
        print("--Loading PDF--")
        # file_path = r"C:\Users\User\Desktop\25052025\ATB Bank Nigeria Groq v2.pdf"
        # file_path = static('pdfs/sample.pdf')
        file_path = os.path.join(settings.MEDIA_ROOT, 'pdfs', 'ATB Bank Nigeria Groq v2.pdf')
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = text_splitter.split_documents(documents)
            vector_store.add_documents(documents=all_splits)
            # retrieved_docs = vector_store.similarity_search(state["messages"][-1].content)
            retrieved_docs = vector_store.similarity_search(user_input)
            print ("The Print the Message:Joor",state["messages"][-1].content)
            return {"pdf_content": retrieved_docs}
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return {"pdf_content": []}

    def search_web(state: State):
        """Perform web search"""
        try:
            tavily_search = TavilySearchResults(max_results=3)
            # search_docs = tavily_search.invoke(state["messages"][-1].content)
            search_docs = tavily_search.invoke(user_input)
            
            if any(error in str(search_docs) for error in ["ConnectionError", "HTTPSConnectionPool"]):
                return {"web_content": []}
                
            formatted_docs = "\n\n---\n\n".join(
                f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
                for doc in search_docs
            )
            return {"web_content": [formatted_docs]}
        except Exception as e:
            print(f"Web search error: {e}")
            return {"web_content": []}

    def generate_response(state: State):
        """Generate structured response using context"""
        print("--Generating Answer--")
        
        # Prepare context
        pdf_text = "\n".join(doc.page_content for doc in state["pdf_content"])
        web_text = "\n".join(state["web_content"])
        # print ("Anyagde:",state)
        query_answer=state["query_answer"]
        # Use the local_file_contents variable defined above
        file_contents = local_file_contents
        print("The File Content is ",file_contents)
        print ("Ayo",query_answer)
        # context = f"PDF Content:\n{pdf_text}\n\nWeb Content:\n{web_text}\n\nQuery Answer:\n{query_answer}n\nQFile Content:\n{file_contents}"
        file_contents=file_contents
        print ("Ayo",query_answer)
        context = f"PDF Content:\n{pdf_text}\n\nWeb Content:\n{web_text}\n\nQuery Answer:\n{query_answer}n\nQFile Content:\n{file_contents}"
        
        # Prepare prompt template
        greeting = get_time_based_greeting()
        prompt1 = f"""
You are Damilola, the AI assistant for ATB Bank. {greeting}! How may I help you today?

**Instructions:**
1. Provide professional, accurate responses
2. Use the following context:
{context}
3. Structure your response as:
- Answer: Clear response to the question
- Sources: References used
- Channels: Relevant banking channels
- Sentiment: User sentiment score (-2 to +2)

**Question:** {state["messages"][-1].content}
"""
     
        prompt = f"""
    "You are Damilola, the AI-powered virtual assistant and Data Analyst for ATB Bank, dedicated to providing professional, accurate, and courteous customer support. Use the provided context to answer the user's question in a structured JSON format. Ensure responses are polite and use the language of the user.",
    This is the Question: {ayula}.
    The content of the attached is : {file_contents}
    
    Context: {context}
    
    "OutputFormat": 
        "answer": "str",
        "sources": "List[str]",
        "channel": "List[str]",
        "sentiment": "int"
    
    "Definitions": 
        "answer": "A clear and concise response to the user's question.",
        "sources": "List of sources used to generate the response, including PDF content, web search or Query  results .",
        "channel": "The specific channels referenced in the conversation, such as POS, ATM, or Web.",
        "sentiment": "A rating of the user's conversation experience, ranging from -2 (very bad) to +2 (very good)."
    instructions:
    "RoleAndBehavior": 
        "Always introduce yourself politely based on the current time ::{greeting}(e.g., 'Good morning and welcome to ATB Bank. I’m Damilola. How can I assist you today?').",
        "Emojis": "Use emojis to express emotions appropriate for the tone of the user's input."
        "InformationHandling": 
        
            "PDFQueries": "Provide precise answers using the document.",
            "ExternalQueries": "Utilize an internet search tool for up-to-date information.",
            "DatabaseQueries": "Utilize an Sql Query search tool for database or analtyics related information.",
            "UnresolvedIssues": "If an issue can't be resolved, escalate it and inform the user courteously."
            consider the content of the attached and the question when responding.
        ,
        "ComplaintHandling": 
            "Acknowledgment": "Express empathy when responding to complaints.",
            "Acknowledgment": "Use the tools Web and Pdf as guide on resolving customer compplain or inquiry. if customer is still not satified or the required information to resoluve the user issue is not in the tool, inform the customer that the issue would be escaclated to the support .",
            "ResolutionUpdate": "Communicate the action taken to resolve the issue."
        ,
        "CustomerEngagement": 
            "PositiveFeedback": "Thank customers for their kind words.",
            "Apology": "Sincerely apologize for any dissatisfaction."
        ,
        "Closing": "End interactions politely, asking if the user needs further assistance.",
        "Emojis": "Use emojis to express emotions appropriate for the tone of the user's input."
        "Language": "Use the user's preferred language for responses.", 
        "Committment": "Response must indicate thaat the assistant is a member of the company. eg we offer loan",
        "Politeness": "Maintain a polite and professional tone throughout the conversation.",
"""
          

        
        try:
            sys_msg = SystemMessage(content=prompt)
            model_with_structure = model.with_structured_output(Answer)
            response = model_with_structure.invoke([sys_msg] + state["messages"])
            return {
                "answer": response,
                "sources": response.sources,
                "channels": response.channels,
                "sentiment": response.sentiment
            }
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "I apologize, I'm having trouble answering that question.",
                "sources": [],
                "channels": [],
                "sentiment": 0
            }

    def summarize_conversation(state: State):
        """Generate conversation summary"""
        prompt1 = f"""
Summarize this conversation between Damilola (ATB Bank assistant) and a customer:

**Conversation:**
{state["messages"]}

**Summary Requirements:**
1. Key points discussed
2. Unresolved issues (if any)
3. Overall sentiment (-2 to +2)
4. All sources referenced
"""
        prompt = f"""  
        The chatbot is designed to analyze entire conversations:{state["messages"]}  between a virtual assistant and a user, generating a structured JSON summary that captures unresolved concerns, sentiment, and sources referenced in the discussion.

### **Analysis Approach:**  
1. **Identify Key Topics**: Extract areas of interest or complaints mentioned by the user.  
2. **Track Resolution Status**:  
   - Only unresolved issues in the summary should be included in `mchannel`.  
   - Any issue the assistant has fully resolved should NOT  be noted in `mchannel`.  
3. **Sentiment Evaluation**:  
   - Perform sentiment analysis on the entire conversation:{state["messages"]} .  
   - Give higher priority to the last messages sent by the user for an accurate sentiment assessment.  
4. **Source Attribution**: List all references, including external documents, PDFs,  web search, Query  results.

### **Expected Output Format:**  
```json
''
  "summary": "A concise overview of the entire conversation.",
  "mchannel": ["List of only unresolved concerns, complaints, or interests"], the ouput eg [ATM, POS, Web]"
  "msentiment: -2 to +2,  // Sentiment score (-2: very negative, +2: very positive),"
  "msources": ["List of sources referenced throughout the conversation"]"
```

### **Definitions:**  
- **summary**: A clear and concise summary of the conversation.  
- **mchannel**: Specific unresolved complaints or interests (e.g., POS, ATM, Web).  
- **msentiment**: Sentiment rating (-2 to +2), giving higher priority to the user's latest messages.  
- **msources**: List of sources used, including PDFs or web search results.

---
"""


        try:
            model_with_structure = model.with_structured_output(Summary)
            # response = model_with_structure.invoke([sys_msg] +(statement))
            response = model_with_structure.invoke([SystemMessage(content=prompt)]+state["messages"])
                
            # response = model_with_structure.invoke([SystemMessage(content=prompt)])
            return {
                "summary": response.summary,
                "main_channels": response.main_channels,
                "overall_sentiment": response.overall_sentiment,
                "all_sources": response.all_sources
            }
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return {
                "summary": "Unable to generate summary",
                "main_channels": [],
                "overall_sentiment": 0,
                "all_sources": []
            }
            
            
    
    def write_query(state: State):
        """Generate SQL query to fetch information."""
        # input= state["messages"][-1].content
        input= user_input
        user_prompt = "Question: {input}"
        
     

        system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""
        

        query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                # "input": state["messages"][-1].content,
                "input": user_input,
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        resultT = execute_query_tool.invoke(result["query"])
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            # f'Question: {state["messages"][-1].content}\n'
            f'Question: {user_input}\n'
            f'SQL Query: {result}\n'
            f'SQL Result: {resultT}'
        )
        responseY = llm.invoke(prompt)
        print(f"Answer: {responseY.content}")
        return {"query_answer": responseY.content}
        # return {"query": result["query"], "result": resultT, "query_answer": responseY.content}
        
    def prepare_final_output(state: State) -> dict:
        """Prepare the final output response with all conversation data.
        
        Args:
            state: The current conversation state containing messages, answers, and summaries
            
        Returns:
            dict: Contains the formatted response message and all metadata
        """
        try:
            # Extract the latest question and answer
            # current_question = state["messages"][-1].content
            current_question = user_input
            initial_question = state["messages"][0].content
            answer_data = state["answer"]
            print ("IJEBU Answeer", Answer)
            # Create structured Answer object
            response = Answer(
                response=answer_data.response,  # Changed from 'yeild' to 'response'
                sources=state.get("sources", []),
                channels=state.get("channels", []),  # Changed from 'channel' to 'channels'
                sentiment=state.get("sentiment", 0)
            )

            # Prepare summary data
            summary_data = {
                "question": initial_question,
                "answer": response.response,
                "sentiment": response.sentiment,
                "sources": response.sources,
                "channels": response.channels,
                "conversation_summary": state.get("summary", ""),
                "unresolved_channels": state.get("mchannel", []),  # Unresolved issues
                "overall_sentiment": state.get("msentiment", 0),
                "all_sources": state.get("msources", [])
            }

 # For storing list of strings
    
            # For database storage (commented out as in original)
            # from .models import Insight
            # Insight.objects.create(
            #         session_id=1,  # Consider using a dynamic session ID
            #         sentimentAnswer=response.sentiment,
            #         answer=response.response,  # Fixed typo from 'respose' to 'response'
            #         source=json.dumps(response.sources),
            #         ticket=json.dumps(response.channels),
            #         summary=state.get("summary", ""),
            #         sentiment=summary_data["overall_sentiment"],
            #         # mchannel=json.dumps(summary_data["unresolved_channels"]),
            #         # msources=json.dumps(summary_data["all_sources"]),
            #         created_at=datetime.datetime.now(),
            #         question=initial_question,
            #         username=username
            #     )
            
            
            
        
            """Create a new insight if session_id doesn't exist, otherwise update the existing entry."""
            session_id= session
            insight, created = Insight.objects.get_or_create(session_id=session_id)

                # Update only if the entry already exists
            if not created:
                    insight.sentimentAnswer = response.sentiment
                    insight.answer = response.response  # Fixed typo from 'respose' to 'response'
                    insight.source = json.dumps(response.sources)
                    insight.ticket = json.dumps(response.channels)
                    insight.summary = state.get("summary", "")
                    insight.sentiment = summary_data["overall_sentiment"]
                    # insight.mchannel = json.dumps(summary_data["unresolved_channels"])
                    # insight.msources = json.dumps(summary_data["all_sources"])
                    insight.updated_at = datetime.datetime.now()  # Update timestamp
                    insight.question = initial_question
                    insight.username = username
                    insight.save()
                    # return f"Insight updated successfully for session {session_id}"

            else:
                    # Creating a new insight entry
                    insight.sentimentAnswer = response.sentiment
                    insight.answer = response.response
                    insight.source = json.dumps(response.sources)
                    insight.ticket = json.dumps(response.channels)
                    insight.summary = state.get("summary", "")
                    insight.sentiment = summary_data["overall_sentiment"]
                    # insight.mchannel = json.dumps(summary_data["unresolved_channels"])
                    # insight.msources = json.dumps(summary_data["all_sources"])
                    insight.created_at = datetime.datetime.now()
                    insight.question = initial_question
                    insight.username = username
                    insight.save()
                  
        
            return {
                "messages": response.response,
                "metadata": summary_data  # Include all structured data
            }

        except KeyError as e:
            print(f"Missing expected state key: {e}")
            return {
                "messages": "I encountered an error processing your request.",
                "metadata": {"error": str(e)}
            }
        except Exception as e:
            print(f"Unexpected error in prepare_final_output: {e}")
            return {
                "messages": "I'm having trouble completing this request.",
                "metadata": {"error": "Internal processing error"}
            }



    def prepare_final_output22(state: State) -> dict:
        """Prepare the final output response with all conversation data.
        
        Args:
            state: The current conversation state containing messages, answers, and summaries
            
        Returns:
            dict: Contains the formatted response message and all metadata
        """
        try:
            # Extract the latest question and answer
            # current_question = state["messages"][-1].content
            current_question =user_input
            initial_question = state["messages"][0].content
            answer_data = state["answer"]
            
            # Create structured Answer object
            response = Answer(
                response=answer_data.response,  # Changed from 'yeild' to 'response'
                sources=state.get("sources", []),
                channels=state.get("channels", []),  # Changed from 'channel' to 'channels'
                sentiment=state.get("sentiment", 0)
            )

            # Prepare summary data
            summary_data = {
                "question": initial_question,
                "answer": response.response,
                "sentiment": response.sentiment,
                "sources": response.sources,
                "channels": response.channels,
                "conversation_summary": state.get("summary", ""),
                "unresolved_channels": state.get("mchannel", []),  # Unresolved issues
                "overall_sentiment": state.get("msentiment", 0),
                "all_sources": state.get("msources", [])
            }

 # For storing list of strings
    
            # For database storage (commented out as in original)
            from .models import Insight

            Insight.objects.create(
                    session_id=1,  # Consider using a dynamic session ID
                    sentimentAnswer=response.sentiment,
                    answer=response.response,  # Fixed typo from 'respose' to 'response'
                    source=json.dumps(response.sources),
                    ticket=json.dumps(response.channels),
                    summary=state.get("summary", ""),
                    sentiment=summary_data["overall_sentiment"],
                    # mchannel=json.dumps(summary_data["unresolved_channels"]),
                    # msources=json.dumps(summary_data["all_sources"]),
                    created_at=datetime.datetime.now(),
                    question=initial_question,
                )
            return {
                "messages": response.response,
                "metadata": summary_data  # Include all structured data
            }

        except KeyError as e:
            print(f"Missing expected state key: {e}")
            return {
                "messages": "I encountered an error processing your request.",
                "metadata": {"error": str(e)}
            }
        except Exception as e:
            print(f"Unexpected error in prepare_final_output: {e}")
            return {
                "messages": "I'm having trouble completing this request.",
                "metadata": {"error": "Internal processing error"}
            }




    def prepare_final_output1(state: State):
        """Prepare final output for response"""
        question=state["messages"][-1].content
        answer = state["answer"]
       
        #

        
        
        return {"messages": state["answer"].response}

    # Setup workflow
    DB_URI = "postgresql://postgres:postgres-user-password@75.119.151.28:5432/postgres?connect_timeout=10"
    
    with PostgresSaver.from_conn_string(DB_URI) as memory:
        memory.setup()
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("search_web", search_web)
        workflow.add_node("search_document", load_and_search_pdf)
        workflow.add_node("write_query", write_query)
        workflow.add_node("generate_answer", generate_response)
        workflow.add_node("summarize", summarize_conversation)
        workflow.add_node("final", prepare_final_output)

        # Define workflow
        workflow.add_edge(START, "search_document")
        workflow.add_edge(START, "search_web")
        workflow.add_edge(START, "write_query")
        workflow.add_edge("search_document", "generate_answer")
        workflow.add_edge("search_web", "generate_answer")
        workflow.add_edge("write_query", "generate_answer")
        workflow.add_edge("generate_answer", "summarize")
        workflow.add_edge("summarize", "final")
        workflow.add_edge("final", END)

        # Execute workflow
        graph = workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "216520"}}
        input_message = HumanMessage(content=message)
        output = graph.invoke({"messages": [input_message]}, config)
        
        return {
            "messages": output["messages"][-1].content,
            "metadata": output.get("metadata", {})
        }



