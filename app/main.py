from openai import OpenAI
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import base64
import re

from langchain.agents import create_sql_agent
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

from nemoguardrails import LLMRails, RailsConfig

load_dotenv()

# imports for LangGraph persistence layer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, MessagesState, StateGraph
from psycopg import Connection

with open("app/bill_explanation.txt", "r", encoding="utf-8") as f:
    bill_explanation_text = f.read()

# Database & environment setup
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URI = os.getenv("DB_URI")

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

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    verbose=True
)

db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=db_toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor.handle_parsing_errors = True


# CSS imeplementation for Streamlit app styling

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("app/style.css")

def get_base64(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def add_logo():
    img_base64 = get_base64("app/pic/digital_bouncers_transparent.png")
    st.markdown(
        f"""
        <style>
        .custom-logo-container {{
            position: fixed;
            top: 40px;
            left: 70px;
            z-index: 1000;
        }}
        .custom-logo-container img {{
            width: 250px;  /* Make logo bigger */
            height: auto;
        }}
        </style>
        <div class="custom-logo-container">
            <img src="data:image/png;base64,{img_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )

add_logo()


st.title("Digital Bouncers Smart Home Assistant")


# UI sidebar guidelines & description component 

with st.sidebar:
    st.markdown("## What does this smart home assistant do?")
    st.markdown(
        """
        This assistant has been trained on your home's past year of energy consumption data. You can ask about your historical
        consumption patterns, recommendations on how to save energy, as well as other energy related inquiries like utility
        bills.
        
        Example prompts:
        - *Which part of the house consumed the most energy in August?*
        - *What recommendations do you have to save energy in the kitchen?*
        - *Please explain what T&D is on my utility bill.*
        """
    )
    st.markdown("## Guidelines")
    st.markdown(
        """
        - Ask questions which are relevant to your home's energy consumption.
        - Please limit your requests to 1 question per message.
        - Be polite, and do not include anything inappropriate or offensive in your message.
        - If any of these are violated, the assistant will not be able to answer your request.
        """
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def text_to_sql(state: MessagesState):

    system_prompt = (
       "You are a helpful assistant helping to generate SQL queries to answer the homeowner's question about energy use. Below is the homeowner's question, followed by context from the past three interactions."
    )
    
    # construct persistent prompt
    full_prompt = [SystemMessage(content=system_prompt)] + state['messages'][-4:][::-1]

    result = agent_executor.invoke(full_prompt, handle_parsing_errors=True)['output']
    print("********************************")
    print(full_prompt)
    print("********************************")
    return {"messages": result}


def classify_bill_question(question: str) -> str: 
    classification_prompt = f"""
You are a strict classification assistant. Decide if the user is asking to explain a component of their utility bill. 
General questions about energy savings / efficiency are not about utility bill explanation.

If yes, respond EXACTLY with:
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


def generate_rag_answer(state: MessagesState):

    system_prompt = (
       "You are a helpful assistant helping the homeowner understand utility bills. Below is the homeowner's question, followed by context from the past three interactions"
    )
    
    # construct persistent prompt
    full_prompt = [SystemMessage(content=system_prompt)] + state['messages'][-4:][::-1]

    print(full_prompt)

    result = llm.invoke(full_prompt)
    return {"messages": result}

def general_answer(state: MessagesState):

    system_prompt = (
       "You are a helpful assistant helping the homeowner answer general questions about energy usage, savings, and efficiency. Below is the homeowner's question, followed by context from the past three interactions"
    )
    
    # construct persistent prompt
    full_prompt = [SystemMessage(content=system_prompt)] + state['messages'][-4:][::-1]

    print(full_prompt)

    result = llm.invoke(full_prompt)
    return {"messages": result}


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


# set up LangGraph for persistence layer
rag_workflow = StateGraph(state_schema=MessagesState)
sql_workflow = StateGraph(state_schema=MessagesState)
gen_workflow = StateGraph(state_schema=MessagesState)

# nodes and edges for the RAG workflow
rag_workflow.add_node("model", generate_rag_answer)
rag_workflow.add_edge(START, "model")

# nodes and edges for the SQL workflow
sql_workflow.add_node("model", text_to_sql)
sql_workflow.add_edge(START, "model")

# notes and edges for the GEN workflow
gen_workflow.add_node("model", general_answer)
gen_workflow.add_edge(START, "model")

# initiate thread_ids
sql_thread = 44
rag_thread = 45
gen_thread = 46

if "curr_thread" not in st.session_state:
    st.session_state.curr_thread = sql_thread


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    to_answer = True
    to_sql = False
    recall = False
    rag = False
    random = False

    config = RailsConfig.from_path("config")  # your existing Nemo config folder
    rails = LLMRails(config)

    with st.chat_message("user"):
        st.markdown(prompt)

        if prompt:
            check_input = rails.generate(messages=[{"role": "user", "content": prompt}])

            print("Nemo moderation output:", check_input)
            print(f"Pass input moderation: {check_input['content']}")
            # if check_input['content'] == "I'm sorry, I can't respond to that.":
            if check_input['content'] == 'No':
                st.markdown("Refusing to answer")
                to_answer = False

            elif check_input['content'] == 'SQL':
                to_sql = True

            elif check_input['content'] == 'Recall':
                recall = True

            elif check_input['content'] == 'RAG':
                rag = True

            elif check_input['content'] == 'Random':
                random = True

            print("##########################")
            print(rails.explain().print_llm_calls_summary())
            print("##########################")

            st.markdown(f"NeMo input moderator: {check_input['content']}")                

    with st.chat_message("assistant"):

        if to_answer == False:
            st.markdown("I'm sorry, I can't respond to that.")
            st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I can't respond to that."})

        elif random:
            response = llm.predict(prompt)
            st.markdown(response)

        elif recall:
            with Connection.connect(DB_URI, **db_params) as conn:
                checkpointer = PostgresSaver(conn)

                curr_thread = st.session_state.curr_thread

                print(f"curr_thread: {curr_thread}")

                # compile the workflow with the Postgres checkpointer
                if curr_thread == 44:
                    app = sql_workflow.compile(checkpointer=checkpointer)
                elif curr_thread == 45:
                    app = rag_workflow.compile(checkpointer=checkpointer)
                elif curr_thread == 46:
                    app = gen_workflow.compile(checkpointer=checkpointer)

                config = {"configurable": {"thread_id": str(st.session_state.curr_thread)}}
                input_data = {"messages": [("human", prompt)]}

                response = app.invoke(input_data, config)['messages'][-1]
                                                        
                st.markdown(response.content)
        
        else:
            # If Nemo says "SQL", do text2sql approach
            if to_sql:
                # open Postgres connection to access prior interactions
                with Connection.connect(DB_URI, **db_params) as conn:
                    st.markdown("Initiated Text-2-SQL workflow")

                    checkpointer = PostgresSaver(conn)
                    sql_flow = sql_workflow.compile(checkpointer=checkpointer)

                    # specifiy LangGraph configuration
                    config = {"configurable": {"thread_id": str(sql_thread)}}
                    input_data = {"messages": [("human", prompt)]}

                    # run the RAG persistence flow
                    agent_result = sql_flow.invoke(input_data, config)['messages'][-1]

                    st.markdown(agent_result.content)

                    st.session_state.messages.append({"role": "assistant", "content": agent_result})

                    st.session_state.curr_thread = sql_thread
                    
                    print(f"curr_thread: {st.session_state.curr_thread}")

            elif rag:

                # 1) Classification step
                classification_label = classify_bill_question(prompt)
                # st.write(f"DEBUG: classification_label => {classification_label}")

                if classification_label == "bill_explanation":

                    st.markdown("Initiated RAG workflow")

                    # 2) RAG
                    rag_prompt = f"""
                        You are a helpful assistant. 
                        Use ONLY the following context to answer the user's question. 
                        If the context does not address their question, say "I don't know."

                        --- BEGIN CONTEXT ---
                        {bill_explanation_text}
                        --- END CONTEXT ---

                        User question: {prompt}
                        """
                    
                    # open Postgres connection to access prior interactions
                    with Connection.connect(DB_URI, **db_params) as conn:
                        checkpointer = PostgresSaver(conn)
                        rag_flow = rag_workflow.compile(checkpointer=checkpointer)

                        # specifiy LangGraph configuration
                        config = {"configurable": {"thread_id": str(rag_thread)}}
                        input_data = {"messages": [("human", rag_prompt)]}

                        # run the RAG persistence flow
                        draft_answer = rag_flow.invoke(input_data, config)['messages'][-1].content
                        # print("Workflow response:", draft_answer)


                    # st.write(f"DEBUG: RAG draft => {draft_answer}")

                    # 3) RAG_eval
                    eval_result = evaluate_rag_answer(prompt, bill_explanation_text, draft_answer)
                    # st.write(f"DEBUG: RAG Eval => {eval_result}")

                    # If "ANSWER OK", keep draft. If "REVISE ANSWER:", keep that
                    if eval_result.startswith("ANSWER OK"):
                        st.markdown("RAG Response Accepted")
                        final_answer = draft_answer

                    elif eval_result.startswith("REVISE ANSWER:"):
                        st.markdown("RAG Response Revised")
                        final_answer = eval_result
                        
                    else:
                        st.markdown("RAG Response Accepted")
                        final_answer = draft_answer  # fallback

                    st.markdown(final_answer)
                
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                else:
                    # Not a bill explanation => fallback to normal approach
                    st.markdown("Initiated fallback mechanism")
                    fallback_answer = llm.predict(prompt)
                    st.markdown(fallback_answer)
                    st.session_state.messages.append({"role": "assistant", "content": fallback_answer})

                st.session_state.curr_thread = rag_thread
                print(f"curr_thread: {st.session_state.curr_thread}")

            else:
                st.markdown("Initiated general Q&A mechanism")

                # open Postgres connection to access prior interactions
                with Connection.connect(DB_URI, **db_params) as conn:
                    checkpointer = PostgresSaver(conn)
                    gen_flow = gen_workflow.compile(checkpointer=checkpointer)

                    # specifiy LangGraph configuration
                    config = {"configurable": {"thread_id": str(gen_thread)}}
                    input_data = {"messages": [("human", prompt)]}

                    # run the RAG persistence flow
                    answer = gen_flow.invoke(input_data, config)['messages'][-1]

                    st.markdown(answer.content)

                st.session_state.messages.append({"role": "assistant", "content": answer})

                st.session_state.curr_thread = gen_thread
                print(f"curr_thread: {st.session_state.curr_thread}")
