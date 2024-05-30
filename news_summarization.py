# To use this script: streamlit run news_summarization.py 


from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool
)
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
import streamlit as st
from email.message import EmailMessage

import os, getpass, ssl, smtplib
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")
        
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("SERPER_API_KEY")

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_PROJECT"] = "CrewAI - News search and monitoring"

email_sender = 'hiddenkirby@gmail.com'
email_password = os.environ.get('PYTHON_GMAIL_KEY')
#email_recipients = ['skuo@simatree1.com','rkirby@simatree1.com','jhu@simatree1.com']
email_recipients = ['rkirby@simatree1.com']


# Initialize the tools
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# gpt-4-turbo - latest, most expensive
open_ai_llm = ChatOpenAI(model_name="gpt-4-turbo")

# local LLM
local_llm = ChatOpenAI(
    openai_api_base="http://192.168.1.158:11434/v1",
    openai_api_key="ollama",                 
    model_name="llama3"
)

industries = ["Financial Services", "Banking", "Real Estate", "Energy", "Entertainment",
                  "Media and Communications", "Retail", "Pharmaceuticals", "Biotech", 
                  "Health Care", "Food Products", "Transportation"]

title = 'AI Research & Reporting'

st.title(title)
use_local_llm = st.checkbox('Use local LLM (Free and Private)')
selected_industry = st.selectbox('Industry', industries)
if use_local_llm:
    st.write('Using Local LLM')
else: 
    st.write('Using OpenAI API Key')

task_description = st.text_area(
    'Task Description',
    f"""Search and filter the latest financial news across various platforms to identify articles and reports indicating major news events in the {selected_industry} industry.
    Focus on news related to severe risks to audit quality, alleged fraud, restatements, or bankruptcy.
    Summarize key findings and flag any articles that require further detailed analysis.
    """,
    height=200
)
task_expected_output = st.text_area(
    'Expected Output',
    """A compiled list of any financial news identified as major news.
    Do not include any PCAOB reports.
    Do not include any reports older than three months ago.
    For each general news topic found:
    - Summarize the major news.
    - Include any links to original articles.
    - Flag the news by type - fraud, bankruptcy, restatement, or severe audit risks.
    """,
    height=200
)

if use_local_llm:
    llm = local_llm
else:
    llm = open_ai_llm 
    
writer = Agent(
    role="Professional Email Writer",
    goal=f"Create clear, concise, and polite emails for various business contexts, ensuring effective communication with clients, partners, and internal teams.",
    backstory="This agent brings a wealth of experience in business communication, honed through years of working in corporate environments where effective communication is critical.",
    verbose=True,
    llm=llm
)

task_write = Task(
    description="Craft and send a professional email.",
    expected_output=f"""Provide a draft template of an email containing the report output.
    Include the title of the report as "Major News Events for {selected_industry}."
    """,
    agent=writer
)

if 'clicked' not in st.session_state:
    st.session_state.clicked = False
    
def click_button():
    st.session_state.clicked = True

def resset_button():
    st.session_state.clicked = False
    
button_click = st.button("Initiate Crew", on_click=click_button, disabled=st.session_state.clicked)

if st.session_state.clicked and task_description is not None and task_expected_output is not None:
    st.write('Crew Initiated.')
    st.write('Running Task...')
    # Define the agents
    researcher = Agent(
        role='Financial News Analyst Agent',
        goal=f'To search the internet and financial news websites and report recent news related to severe risks to audit quality, fraud, restatement or bankruptcy in the {selected_industry} industry.',
        backstory=f"A highly experienced analyst with a keen eye for details that could impact {selected_industry} market and investor decisions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool, web_rag_tool, YahooFinanceNewsTool()],
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
        'CrewAI Email Draft',
        result,
        height=800 
    )
    st.button('Start Over', on_click=resset_button)
    
    # (optional) send email
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = ", ".join(email_recipients)
    em['Subject'] = title
    em.set_content(result)
    
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_recipients, em.as_string())
else: 
    st.warning('Please provide a task description and expected output. ')
