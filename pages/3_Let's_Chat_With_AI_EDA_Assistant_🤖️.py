import pandas as pd

import os

from langchain_experimental.agents import create_pandas_dataframe_agent
from apikey import apikey 
from dotenv import load_dotenv, find_dotenv

import streamlit as st

from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

#OpenAIKey
os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())

st.title("Let's Chat With AI EDA Assistant")

st.markdown(
    """

<p> This is an AI EDA Assistant utilized OpenAI API, please feel free to ask him about any EDA questions.🤖️</p>

<p> Step 1: Select variables you would like to explore.</p>

<p> Step 2: Select questions from the list below or select others to ask your own questions (e.g. ask him what can you do? tell me the correlations between these variables.)</p>

<p> Let't get started! </p>
""",

    unsafe_allow_html=True
)

data_file = 'brfss_cleaned.csv'
data = pd.read_csv(data_file)

# Step3: Select X and Y
def input(data):
    selected_columns = st.multiselect("Select at least one variable you would like to explore", data.columns)
    df = data[selected_columns]

    return df


# Generate LLM response
def generate_response(df, input_query):
  llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2)
  # Create Pandas DataFrame Agent
  agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
  # Perform Query using the Agent
  response = agent.run(input_query)
  return st.write(response)


df = input(data)

question_list = [
  'How many rows are there?',
  'How many columns are there',
  'Other']
query_text = st.selectbox('Select an example query:', question_list)



if len(df.columns) > 0:
    if query_text is 'Other':
        query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...')
    st.header('Output')
    generate_response(df, query_text)



