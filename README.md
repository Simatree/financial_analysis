# financial_analysis

CrewAI scripts tailored to financial news reporting, alerting, and analysis

## developer setup

1.  (running Python 3.12.2)
2.  run `python3 -m venv .venv`
3.  run `source .venv/bin/activate`
4.  run `pip install -U -r requirements.txt`

# ollama on docker setup (local dev)

1. ensure docker is installed
2. run `docker ps` to check if it's not already running
3. to spin ollama up on a container - run `docker run -d -v /Users/ryankirby/ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama`
4. to interact with ollama client - run `docker exec -it ollama ollama run llama2` - to run llama2 LLM specifically
5. (optional): add an alias to your .bashrc file by adding `alias ollama='docker exec -it ollama ollama'`
6. (optional): then you can run `ollama run llama2` like normal

# run apps

1. `python company_financial_analysis.py`
2. `python news_summarization.py`
3. `python file_summarization.py`
