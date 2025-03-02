from openai import OpenAI
import streamlit as st
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

from langchain.agents import create_sql_agent
from langchain.utilities import SQLDatabase 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

from nemoguardrails import LLMRails, RailsConfig
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore
from typing import List

load_dotenv()

with open("app/bill_explanation.txt", "r", encoding="utf-8") as f:
    bill_explanation_text = f.read()

# database & environment set up
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("API_KEY")

db_params = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": "5432"
}
db = SQLDatabase.from_uri(
    f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["dbname"]}',
    include_tables=["smart_home_data"]
)

client = OpenAI(api_key=OPENAI_API_KEY)

# LLM set up
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    verbose=True
)

# Text-2-SQL Agent executor
db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=db_toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor.handle_parsing_errors = True

# app title
st.title("Digital Bouncers Smart Home Assistant")

# initialize LangGraph's memory for conversation tracking
conversation_memory = InMemoryStore()

def classify_bill_question(question: str) -> str: 

    classification_prompt = f"""
        You are a strict classification assistant. Decide if the user is asking
        for an electricity bill explanation. If yes, respond EXACTLY with:
        bill_explanation
        If no, respond EXACTLY with:
        not_bill_explanation

        User question: {question}
        """
    
    result = llm.predict(classification_prompt).strip().lower()
    # Enforce only two possible labels
    if "bill_explanation" in result:
        return "bill_explanation"
    
    return "not_bill_explanation"

def generate_rag_answer(question: str, context: str) -> str:
    
    rag_prompt = f"""
        You are a helpful assistant. 
        Use ONLY the following context to answer the user's question. 
        If the context does not address their question, say "I don't know."

        --- BEGIN CONTEXT ---
        {context}
        --- END CONTEXT ---

        User question: {question}
        """
    
    return llm.predict(rag_prompt)

def evaluate_rag_answer(question: str, context: str, draft_answer: str) -> str:
    
    eval_prompt = f"""
        The user asked: {question}
        We have this reference context:
        {context}

        The assistant's draft answer:
        {draft_answer}

        Check if the answer is correct and consistent with the reference context.
        If correct, respond with "ANSWER OK".
        If there's a factual issue or it's incomplete, respond with 
        "REVISE ANSWER: <short correction>".

        Do not add extra text beyond these formats.
        """

    result = llm.predict(eval_prompt).strip()
    return result


class MessagesState(MessagesState):
    """Defines the state structure for LangGraph workflow."""
    user_prompt: str
    history: List[dict]  # List of past messages (prompt-response pairs)
    response: str

# define LangGraph nodes and execution flow
def start(state: MessagesState):
    """"Retrieve past interactions"""
    history = conversation_memory.memories
    return MessagesState(user_prompt= state["user_prompt"], history= history, response="")
    
def process(state: MessagesState):
    """Determine if query is Text-2-SQL, RAG, or fallback"""
    user_prompt = state["user_prompt"]
    
    config = RailsConfig.from_path("config")
    rails = LLMRails(config)

    check_input = rails.generate(messages=[{"role": "user", "content": user_prompt}])

    if check_input['content'] == "I'm sorry, I can't respond to that.":
        return MessagesState(user_prompt=user_prompt, history=state.history, response="I'm sorry, I can't respond to that.")
    
    if check_input['content'] == 'SQL':
        agent_result = agent_executor.invoke(user_prompt, handle_parsing_errors=True)
        return MessagesState(user_prompt=user_prompt, history=state.history, response=agent_result)
    
    # check for RAG-eligibility
    classification_label = classify_bill_question(user_prompt)

    if classification_label == "bill_explanation":
        draft_answer = generate_rag_answer(user_prompt, bill_explanation_text)
        eval_result = evaluate_rag_answer(user_prompt, bill_explanation_text, draft_answer)

        if eval_result.startswith("ANSWER OK"):
            final_answer = draft_answer
        elif eval_result.startswith("REVISE ANSWER:"):
            final_answer = eval_result
        else:
            final_answer = draft_answer  # Fallback

        return MessagesState(user_prompt=user_prompt, history=state.history, response=final_answer)
    
    # default fallback
    fallback_answer = llm.predict(user_prompt)
    return MessagesState(user_prompt=user_prompt, history=state.history, response=fallback_answer)
    
def store(state: MessagesState):
    """Store the conversation in LangGraph memory"""
    conversation_memory.add_memory({
        "user_prompt": state["user_prompt"],
        "response": state["response"]
    })

    return state  # pass context for further usage when needed
    
# define LangGraph's persistence layer
workflow = StateGraph(state_schema=MessagesState)

workflow.add_node("start", start)
workflow.add_node("process", process)
workflow.add_node("store", store)

workflow.add_edge(START, "process")
workflow.add_edge("process", "store")

# Streamlit UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # run LangGraph workflow
    graph_executor = workflow.compile()
    response_context = graph_executor.invoke({
        "user_prompt": prompt, 
        "history": conversation_memory.memories, 
        "response": ""
        })

    with st.chat_message("assistant"):
        st.markdown(response_context["response"])
        st.session_state.messages.append({"role": "assistant", "content": response_context["response"]})
