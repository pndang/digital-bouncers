from openai import OpenAI
import streamlit as st
import psycopg2
import pandas as pd
import os

from langchain.agents import create_sql_agent
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

from nemoguardrails import LLMRails, RailsConfig

# Load secrets from env vars
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("API_KEY")
DB_URI = os.getenv("DB_URI")

# postgreSQL database connection params
db_params = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": "5432"
}

# create the database connection
db = SQLDatabase.from_uri(DB_URI)

# define SQL query
query = """
SELECT * FROM smart_home_data
"""

with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cur:
        cur.execute(query)
        data = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]

df = pd.DataFrame(data, columns=column_names)

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
    with st.chat_message("user"):
        st.markdown(prompt)
        
        # input moderation guardrails
        if prompt:
            print(prompt)
            
            import openai
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            openai.api_key = OPENAI_API_KEY  

            # Load NeMo Guardrails config
            config = RailsConfig.from_path("config")
            rails = LLMRails(config)
            
            check_input = rails.generate(messages=[{
                "role": "user",
                "content": prompt
            }])

            print(f"Pass input moderation: {check_input['content']}")

            st.markdown(f"Pass input moderation: {check_input['content']}")

    with st.chat_message("assistant"):
        pass
        # response = agent_executor.run(prompt)  
        ### Put output guardrail here?
        # st.markdown(response)

    # st.session_state.messages.append({"role": "assistant", "content": response})
A