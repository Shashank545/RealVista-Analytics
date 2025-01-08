print("Chat with Multiple Structured Data in independent manner with both Langchain and PandasAI \n\n\n\n")

import streamlit as st
import pandas as pd
import os, httpx, ssl, warnings, requests, time
from functools import partial
import openai
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from pandasai.llm import OpenAI as poai
from pandasai import SmartDataframe
from pandasai.exceptions import MaliciousQueryError, NoResultFoundError
from PIL import Image
from dotenv import load_dotenv
import warnings




# Patch SSL globally for `httpx`
class UnsafeClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)

httpx._default_client = UnsafeClient()



############################################################# Handling SSL Error ###############################################################

# Load ALL environment variables
load_dotenv()

# OpenAI API key setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai._default_api_client = UnsafeClient()
openai.debug = True

# OpenAI Client Imports
# from openai import OpenAI
# opai_client = OpenAI(api_key=OPENAI_API_KEY)
openai.api_key = OPENAI_API_KEY



GENAI_MODEL_A = "gpt-3.5-turbo-1106"
GENAI_MODEL_B = "gpt-4o-2024-08-06"
GENAI_MODEL_C = "gpt-4o-mini-2024-07-18"
GENAI_MODEL_D = "gpt-4o-mini-2024-07-18"


# Streamlit app configuration
st.set_page_config(page_title="Smart Chat with Dataframe", page_icon="🤖")

st.title("🤖 Smart Chat with Dataframe")

# CSS styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e3f2fd, #c9eaf3);
    }
    </style>
""", unsafe_allow_html=True)

# File selection: Default or Upload
st.sidebar.title("Configuration")
data_source = st.sidebar.radio(
    "Choose Data Source:",
    ("Use Default Data", "Upload Custom Data")
)

# Load data based on selection
if data_source == "Use Default Data":
    # Load default dataset
    default_file_path = "data/transact.csv"  # Replace with your actual file path
    if os.path.exists(default_file_path):
        df = pd.read_csv(default_file_path)
        st.sidebar.success("Loaded default data.")
    else:
        st.sidebar.error("Default data file not found. Please upload a custom file.")
        df = None
elif data_source == "Upload Custom Data":
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        if file_ext in [".csv"]:
            df = pd.read_csv(uploaded_file)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            df = None
    else:
        st.warning("Please upload a file to proceed.")
        df = None

# Display dataset info and preview
if df is not None:
    st.write("### Dataset Summary")
    st.markdown(""" This dataset provides comprehensive information on various types of real estate transactions in Dubai, including property sales, 
                    rentals, and transfers. It covers a wide range of transaction details, such as property type, usage, registration, area, building 
                    and project information, nearest landmarks, facilities, and transaction values. """)
    col1, col2 = st.columns([3, 5])

    with col1:
        st.metric("Number of Rows", df.shape[0])
        st.metric("Number of Columns", df.shape[1])

    with col2:
        st.write("**Available Columns:**")
        st.write(" , ".join(df.columns))
    # st.write("### Data Preview")
    # st.write(df.head())
    st.write("### Usage Ideas")
    st.markdown(""" The dataset can be used for various purposes, including:
                    Analyzing the real estate market trends in Dubai.
                    Identifying patterns and relationships between property types, locations, and transaction values.
                    Evaluating the impact of government policies and regulations on the real estate market.
                    Assessing market demand and supply for various property types and locations.
                    Developing predictive models for real estate prices and rental rates.""")
else:
    st.stop()

# OpenAI API Key input
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.sidebar.warning("Please enter your OpenAI API Key to proceed.")

# Classify intent using OpenAI
def classify_intent(query, openai_api_key):
    system_prompt = (
        "You are an AI assistant specialized in classifying user queries into distinct analytical categories. "
        "Your task is to identify the most relevant category based on the user's input. "
        "The categories are:\n"
        "1. Transactional analysis\n"
        "2. Comparative analysis\n"
        "3. Descriptive analysis\n"
        "4. Predictive analysis\n"
        "5. Visual analysis\n"
        "6. Reasoning\n"
        "7. Others\n"
        "Provide a one-word response corresponding to the category name."
    )

    import requests

    try:
        # Define payload and headers
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "max_tokens": 10,
            "temperature": 0
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_api_key}'  # Replace with your actual API key
        }

        # Make API request
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, verify=False)

        print(f"HTTPS Code : {response.status_code}")

        # Process response
        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
            print(response_text)
            return response_text
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error in classifying intent: {str(e)}")



# Main chatbot functionality
st.write("### Ask your Query")
prompt = st.text_area("Enter your query:", placeholder="Ask a question about the uploaded data.")

if st.button("Analyze"):
    if not openai_api_key:
        st.error("OpenAI API Key is required.")
        st.stop()

    if not prompt.strip():
        st.warning("Please enter a query.")
        st.stop()

    # Classify user intent
    with st.spinner("Classifying query intent..."):
        intent = classify_intent(prompt, openai_api_key)

    st.write(f"**Intent Detected:** {intent.capitalize()}")

    if intent == "Others":
        llm = ChatOpenAI(model=GENAI_MODEL_C, openai_api_key=openai_api_key)
        
        # Wrap agent creation in a try-except block
        try:
            pandas_df_agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                # return_intermediate_steps=True,
                # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                # agent_type="openai-tools",
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
        except openai.APIConnectionError as e:
            st.error(f"API Connection Error occurred: {e}")


        with st.spinner("Analyzing with Langchain..."):
            retries = 3
            # print(pandas_df_agent.invoke(prompt))
            for attempt in range(retries):
                try:
                    # response = pandas_df_agent.run(prompt)
                    response = pandas_df_agent.invoke(
                                            {
                                                "input": prompt
                                            })
                    st.write("### Analysis Response")
                    st.write(response["output"])
                    break
                except openai.APIConnectionError as e:
                    if attempt < retries - 1:
                        st.warning(f"Attempt {attempt + 1} failed. Retrying...")
                        time.sleep(5)
                    else:
                        st.error("Failed to connect to OpenAI API after multiple attempts.")

            
    elif intent in ["Transactional analysis", "Visual analysis", "Comparative analysis", "Descriptive analysis", "Predictive analysis", "Reasoning"]:
        model = poai(api_token=openai_api_key)
        smart_df = SmartDataframe(df, config={"llm": model})

        if intent == "Visual analysis":
            with st.spinner("Generating visualization with PandasAI..."):
                try:
                    response = smart_df.chat(prompt)
                    if os.path.isfile(response) and response.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image = Image.open(response)
                        st.image(image, caption="Generated Chart", use_container_width=True)
                    else:
                        st.write("### Visualization Result")
                        st.write(response)
                except MaliciousQueryError:
                    st.error("Malicious query detected. Please rephrase.")
                except NoResultFoundError:
                    st.warning("No results found. Try rephrasing your query.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            response = smart_df.chat(prompt)
            st.write(f"### {intent} Result")
            st.write(response)
    else:
        st.error("Unrecognized intent. Please refine your query.")
