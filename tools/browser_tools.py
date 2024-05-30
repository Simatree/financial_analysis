from requests_html import HTMLSession
from langchain.tools import tool
from crewai import Agent, Task
from unstructured.partition.html import partition_html
import streamlit as st

class BrowserTools:

    @tool("Scrape website content")
    def scrape_and_summarize_website(website):
        """Useful to scrape and summarize a website content"""
        session = HTMLSession()
        response = session.get(website)
        
        # Render the page, in case there's JavaScript dynamically rendering elements
        response.html.render()

        # Use partition_html on the HTML, not on the text
        elements = partition_html(html=response.html.html)
        
        # Joining the extracted elements into a single string, then splitting into chunks
        content = "\n\n".join([str(el) for el in elements])
        content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
        summaries = []

        for chunk in content:
            agent = Agent(
                role='Principal Researcher',
                goal='Do amazing research and summaries based on the content you are working with',
                backstory="You're a Principal Researcher at a big company and you need to do research about a given topic.",
                allow_delegation=False
            )
            task = Task(
                agent=agent,
                description=f'Analyze and summarize the content below, make sure to include the most relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}'
            )
            summary = task.execute()
            summaries.append(summary)

        st.write('Scraping Website to memory {}... '.format(website))
        return "\n\n".join(summaries)
