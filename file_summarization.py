# To use this script: streamlit run file_summarization.py 


from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
import streamlit as st

import os, getpass
from dotenv import load_dotenv
load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_PROJECT"] = "CrewAI - Ask your Document"

# gpt-4-turbo - latest, most expensive
open_ai_llm = ChatOpenAI(model_name="gpt-4-turbo")

# local LLM
local_llm = ChatOpenAI(
    openai_api_base="http://192.168.1.158:11434/v1",
    openai_api_key="ollama",
    model_name="llama3"
)

title = 'AI Document Content Summarizer'

st.title(title)
use_local_llm = st.checkbox('Use local LLM (Private and Secure)')

if use_local_llm:
    st.write('Using Secure LLM')
else:
    st.write('Using OpenAI')

uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

# Directory where files will be saved
save_directory = "test_data"
os.makedirs(save_directory, exist_ok=True)

# If a file has been uploaded
if uploaded_file is not None:
    # Define the full file path
    save_path = os.path.join(save_directory, uploaded_file.name)

    # Save the uploaded file
    save_uploaded_file(uploaded_file, save_path)

    # Display success message
    st.success(f"File saved successfully at: {save_path}")
else:
    st.info("Upload a file to ask it questions.")

task_description = st.text_area(
    'Task Description',
    f"""Given the following input file, summarize the key findings.
    Flag any items that require further detailed analysis.
    """,
    height=200
)

task_expected_output = st.text_area(
    'Expected Output',
    """A professional written summarization of the input document.
    """,
    height=200
)

if use_local_llm:
    llm = local_llm
else:
    llm = open_ai_llm

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

def resset_button():
    st.session_state.clicked = False

button_click = st.button("Ask it a question", on_click=click_button, disabled=st.session_state.clicked)

if st.session_state.clicked and uploaded_file is not None and task_description is not None and task_expected_output is not None:
    st.write('Initiated.')
    st.write('Running Task...')
    
    # Initialize the tools
    pdf_read_tool = PDFSearchTool(
        pdf=save_path,
    )

    researcher = Agent(
        role='Financial News Analyst Agent',
        goal=f'To search the internet and financial news websites and report news related to severe risks to audit quality, fraud, restatement, or bankruptcy in any industry.',
        backstory=f"A highly experienced analyst with a keen eye for details that could impact market and investor decisions.",
        verbose=True,
        allow_delegation=False,
        tools=[pdf_read_tool],
        llm=llm
    )

    # Define the tasks
    research_task = Task(
        description=task_description,
        expected_output=task_expected_output,
        agent=researcher
    )

    # Instantiate the crew
    document_crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,  # Tasks will be executed one after the other,
        verbose=True
    )
    # Begin the task execution
    result = document_crew.kickoff()
    st.text_area(
        'AI Response',
        result,
        height=800
    )
    st.button('Start Over', on_click=resset_button)
else:
    st.warning('Please provide a task description and expected output, and upload a file.')
