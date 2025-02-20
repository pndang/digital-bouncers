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


load_dotenv()

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
db = SQLDatabase.from_uri(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["dbname"]}', include_tables=["smart_home_data"])

# Define SQL query
query= """
SELECT * FROM smart_home_data
"""

with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cur:
        cur.execute(query)
        data = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]

df = pd.DataFrame(data, columns=column_names) 

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, 
    model_name="gpt-3.5-turbo", 
    temperature=0, 
    max_tokens=500,
    streaming=True
)

db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=db_toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

st.title("Digital Bouncers Smart Home Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        ### Put guardrail here?

    with st.chat_message("assistant"):
        response = agent_executor.run(prompt)  
        ### Put output guardrail here?
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
