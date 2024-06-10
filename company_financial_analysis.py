# To use this script: streamlit run company_financial_analysis.py

import os, getpass
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from textwrap import dedent

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
title = 'AI Company Financial Analyzer'
os.environ["LANGCHAIN_PROJECT"] = title

# gpt-4-turbo - latest, most expensive
open_ai_llm = ChatOpenAI(model_name="gpt-4-turbo")

# local LLM
local_llm = ChatOpenAI(
    openai_api_base="http://192.168.1.158:11434/v1",
    openai_api_key="ollama",
    model_name="phi3:mini"
)

st.title(title)
use_local_llm = st.checkbox('Use local LLM (Private and Secure)')

if use_local_llm:
    llm = local_llm
    st.write('Using Secure LLM')
else:
    llm = open_ai_llm 
    st.write('Using OpenAI')
    
company_name = st.text_input(
    'Company Name',
    'Apple Inc',
    max_chars=100
    )
    
task_description = st.text_area(
    'Task Description',
    f"""Review and synthesize the analyses provided by the
    Financial Analyst and the Research Analyst.
    Combine these insights to form a comprehensive
    investment recommendation.
    """,
    height=200
)
task_expected_output = st.text_area(
    'Expected Output',
    """You MUST Consider all aspects, including financial
    health, market sentiment, and qualitative data from
    EDGAR filings.

    Make sure to include a section that shows insider 
    trading activity, and upcoming events like earnings.

    Your final answer MUST be a recommendation for your
    customer. It should be a full super detailed report, providing a 
    clear investment stance and strategy with supporting evidence.
    Make it pretty and well formatted for your customer.
    """,
    height=300
)

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools
from tools.ExaSearchTool import ExaSearchTool

from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

class StockAnalysisAgents():
    def financial_analyst(self):
        return Agent(
            role='The Best Financial Analyst',
            goal="""Impress all customers with your financial data 
            and market trends analysis""",
            backstory="""The most seasoned financial analyst with 
            lots of expertise in stock market analysis and investment
            strategies that is working for a super important customer.""",
            verbose=True,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                CalculatorTools.calculate,
                SECTools.search_10q,
                SECTools.search_10k,
                ExaSearchTool.search,
                ExaSearchTool.find_similar,
                ExaSearchTool.get_contents
            ],
            llm=llm
        )

    def research_analyst(self):
        return Agent(
            role='Staff Research Analyst',
            goal="""Being the best at gather, interpret data and amaze
            your customer with it""",
            backstory="""Known as the BEST research analyst, you're
            skilled in sifting through news, company announcements, 
            and market sentiments. Now you're working on a super 
            important customer""",
            verbose=True,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                SearchTools.search_news,
                YahooFinanceNewsTool(),
                SECTools.search_10q,
                SECTools.search_10k,
                ExaSearchTool.search,
                ExaSearchTool.find_similar,
                ExaSearchTool.get_contents
            ],
            llm=llm
    )

    def investment_advisor(self):
        return Agent(
            role='Private Investment Advisor',
            goal="""Impress your customers with full analyses over stocks
            and complete investment recommendations""",
            backstory="""You're the most experienced investment advisor
            and you combine various analytical insights to formulate
            strategic investment advice. You are now working for
            a super important customer you need to impress.""",
            verbose=True,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                SearchTools.search_news,
                CalculatorTools.calculate,
                YahooFinanceNewsTool()
            ],
            llm=llm
    )

class StockAnalysisTasks():
  def research(self, agent, companyName):
    return Task(
      description=dedent(f"""
        Collect and summarize recent news articles, press
        releases, and market analyses related to the stock and
        its industry.
        Pay special attention to any significant events, market
        sentiments, and analysts' opinions. Also include upcoming 
        events like earnings and others.
        Selected company by the customer: {companyName}
      """),
      expected_output=dedent("""
      Your final answer MUST be a report that includes a
        comprehensive summary of the latest news, any notable
        shifts in market sentiment, and potential impacts on 
        the stock.
        Also make sure to return the stock ticker.
  
        Make sure to use the most recent data as possible.
      """),
      agent=agent
    )
    
  def financial_analysis(self, agent): 
    return Task(
      description=dedent(f"""
        Conduct a thorough analysis of the stock's financial
        health and market performance. 
        This includes examining key financial metrics such as
        P/E ratio, EPS growth, revenue trends, and 
        debt-to-equity ratio. 
        Also, analyze the stock's performance in comparison 
        to its industry peers and overall market trends.
      """),
      expected_output=dedent("""
      Your final report MUST expand on the summary provided
      but now including a clear assessment of the stock's
      financial standing, its strengths and weaknesses, 
      and how it fares against its competitors in the current
      market scenario.

      Make sure to use the most recent data possible.
      """),
      agent=agent
    )

  def filings_analysis(self, agent):
    return Task(
      description=dedent(f"""
        Analyze the latest 10-Q and 10-K filings from EDGAR for
        the stock in question. 
        Focus on key sections like Management's Discussion and
        Analysis, financial statements, insider trading activity, 
        and any disclosed risks.
        Extract relevant data and insights that could influence
        the stock's future performance.   
      """),
      expected_output=dedent("""
        Your final answer must be an expanded report that now
        also highlights significant findings from these filings,
        including any red flags or positive indicators for
        your customer. 
      """),
      agent=agent
    )
    
#manage refresh button state
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
    
    agents = StockAnalysisAgents()
    tasks = StockAnalysisTasks()

    research_analyst_agent = agents.research_analyst()
    financial_analyst_agent = agents.financial_analyst()
    investment_advisor_agent = agents.investment_advisor()

    print(f"Researching {company_name}...")
    research_task = tasks.research(research_analyst_agent, company_name)
    financial_task = tasks.financial_analysis(financial_analyst_agent)
    filings_task = tasks.filings_analysis(financial_analyst_agent)
    recommend_task = Task(
      description=dedent(task_description),
      expected_output=dedent(task_expected_output),
      agent=investment_advisor_agent
    )
    
    crew = Crew(
        agents=[
            research_analyst_agent,
            financial_analyst_agent,
            investment_advisor_agent
        ],
        tasks=[
            research_task,
            financial_task,
            filings_task,
            recommend_task
        ],
        verbose=2
    )

    # Begin the task execution
    result = crew.kickoff()
    st.text_area(
        'AI Company Financial Analysis Report for ' + company_name,
        result,
        height=800 
    )
    st.button('Start Over', on_click=resset_button)
else: 
    st.warning('Please provide a task description and expected output. ')



       