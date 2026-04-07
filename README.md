## Setting Up the Python Environment with uv
1. **Init project**:
   ```bash
   uv init --python 3.13.5 email_handler-langgraph
   cd email_handler-langgraph
   ```
2. **Venv**
   ```bash
   uv venv
   ```

3. **Dependencies**
   ```bash
   uv add langchain_core langchain-openai langgraph
   uv add ipython
   uv add python-dotenv
   uv add structlog
   ```
