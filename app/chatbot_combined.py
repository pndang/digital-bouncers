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

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("API_KEY")

# postgreSQL database connection params
db_params = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": "5432"
}

# create the database connection
db = SQLDatabase.from_uri(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["dbname"]}', include_tables=["smart_home_data"])

# initialize openai client
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

st.title("Digital Bouncers Smart Home Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    to_answer = True
    with st.chat_message("user"):
        st.markdown(prompt)
        
        # input moderation guardrails
        if prompt:
            # Load NeMo Guardrails config
            config = RailsConfig.from_path("config")
            rails = LLMRails(config)
            
            check_input = rails.generate(messages=[{
                "role": "user",
                "content": prompt
            }])

            print(check_input)
            print(f"Pass input moderation: {check_input['content']}")
            if check_input['content'] == "I'm sorry, I can't respond to that.":
                to_answer = False


            st.markdown(f"Pass input moderation: {check_input['content']}")

    with st.chat_message("assistant"):
        if to_answer:
            print("THIS WAS THE INVOKED PROMPT", agent_executor.invoke(prompt))
            response = agent_executor.invoke(prompt)['output']  
            ### Put output guardrail here?
            st.markdown(response)

    # st.session_state.messages.append({"role": "assistant", "content": response})