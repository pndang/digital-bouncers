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

load_dotenv()

with open("app/bill_explanation.txt", "r", encoding="utf-8") as f:
    bill_explanation_text = f.read()

# Database & environment setup
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

st.title("Digital Bouncers Smart Home Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


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


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    to_answer = True
    to_sql = False

    config = RailsConfig.from_path("config")  # your existing Nemo config folder
    rails = LLMRails(config)

    with st.chat_message("user"):
        st.markdown(prompt)

        if prompt:
            check_input = rails.generate(messages=[{"role": "user", "content": prompt}])

            print("Nemo moderation output:", check_input)
            print(f"Pass input moderation: {check_input['content']}")
            if check_input['content'] == "I'm sorry, I can't respond to that.":
                to_answer = False
            if check_input['content'] == 'SQL':
                to_sql = True
            #print(rails.explain().print_llm_calls_summary())
            #st.markdown(f"Pass input moderation: {check_input['content']}")                

    with st.chat_message("assistant"):
        if to_answer == False:
            st.markdown("I'm sorry, I can't respond to that.")
            st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I can't respond to that."})
        else:
            # If Nemo says "SQL", do text2sql approach
            if to_sql:
                agent_result = agent_executor.invoke(prompt, handle_parsing_errors=True)['output']
                st.markdown(agent_result)
                st.session_state.messages.append({"role": "assistant", "content": agent_result})
            else:
                # 1) Classification step
                classification_label = classify_bill_question(prompt)
                st.write(f"DEBUG: classification_label => {classification_label}")

                if classification_label == "bill_explanation":
                    # 2) RAG
                    draft_answer = generate_rag_answer(prompt, bill_explanation_text)
                    st.write(f"DEBUG: RAG draft => {draft_answer}")

                    # 3) RAG_eval
                    eval_result = evaluate_rag_answer(prompt, bill_explanation_text, draft_answer)
                    st.write(f"DEBUG: RAG Eval => {eval_result}")

                    # If "ANSWER OK", keep draft. If "REVISE ANSWER:", keep that
                    if eval_result.startswith("ANSWER OK"):
                        final_answer = draft_answer
                    elif eval_result.startswith("REVISE ANSWER:"):
                        final_answer = eval_result
                    else:
                        final_answer = draft_answer  # fallback

                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                else:
                    # Not a bill explanation => fallback to normal approach
                    fallback_answer = llm.predict(prompt)
                    st.markdown(fallback_answer)
                    st.session_state.messages.append({"role": "assistant", "content": fallback_answer})
