import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import logging
import re
import subprocess
import time
import yaml
from git import Repo
from typing import List, Dict

import dotenv
from pydantic import BaseModel, Field

# --- LangChain 1.0 Core Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import GitLoader, DirectoryLoader
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- Local Utility Imports (assumed to be correct) ---
try:
    # Assuming these are available in the path
    from recon_agent_new import ReconAgent
    from utils.prompt import PentestAgentPrompt
except ImportError as e:
    print(f"Error importing local utils: {e}. Make sure your PYTHONPATH is set correctly.")
    sys.exit(1)


# --- Configuration and Logging Setup ---
import logging
import sys
import dotenv

dotenv.load_dotenv()

# 1. 获取根日志记录器 (root logger)
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # 设置日志记录的最低级别

# 2. 清除任何可能已存在的处理器，以避免重复输出日志
if logger.hasHandlers():
    logger.handlers.clear()

# 3. 创建一个通用的日志格式，供所有处理器使用
log_formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 4. 创建一个文件处理器 (FileHandler) 并添加到 logger
file_handler = logging.FileHandler('execution_agent_v2.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# 5. 创建一个流处理器 (StreamHandler) 用于输出到控制台，并添加到 logger
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# 6. 获取当前模块的日志记录器，它将继承上面的设置
logger = logging.getLogger(__name__)


try:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    execution_config = config['runtime']['execution']
    logger.info("Configuration loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load configuration: {e}")
    sys.exit(1)


# --- Agent Tools Definition ---
# The agent will have access to these functions. The docstrings are crucial
# as they tell the LLM when and how to use the tool.

def _run_shell_command_logic(command: str, path: str) -> str:
    """
    Executes a shell command in a specified directory and returns its output.
    Use this to run exploit scripts, list files, or interact with the system.
    Only run commands that are safe and necessary for the penetration test.
    The 'path' argument should be the directory where the command needs to be run.
    """
    logger.info(f"Executing command: `{command}` in path: `{path}`")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=path,
            timeout=60
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return output if output.strip() else "Command executed successfully with no output."
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return f"Error: Command failed with exit code {e.returncode}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {str(e)}"

# --- Main Agent Class (Refactored) ---
class ExecutionAgent:
    def __init__(self, doc_dir: str):
        self.doc_dir = doc_dir
        self.llm = self._get_llm()
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent_executor()
        self.memory_store = {} # Store for session histories

    def _get_llm(self):
        """Initializes and returns a LangChain 1.0 model."""
        model_name = execution_config['model']
        logger.info(f"Initializing model: {model_name}")
        # This example uses OpenAI. It can be easily adapted for other providers.
        return ChatOpenAI(model=model_name, temperature=0, streaming=True)

    def _setup_tools(self) -> list:
        """Creates and configures the tools for the agent."""
        # 1. Create a RAG retriever tool to replace the LlamaIndex part
        logger.info("Setting up RAG retriever tool...")
        embeddings = OpenAIEmbeddings()
        # Use a persistent vector store to avoid re-creating it on every run
        vectorstore_path = os.path.join(self.doc_dir, ".faiss_index")
        
        if os.path.exists(vectorstore_path):
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing FAISS vector store from disk.")
        else:
            logger.info(f"Creating new vector store from documents in {self.doc_dir}")
            # Use DirectoryLoader instead of SimpleDirectoryReader
            loader = DirectoryLoader(self.doc_dir, recursive=True, silent_errors=True)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(splits, embeddings)
            vectorstore.save_local(vectorstore_path)
            logger.info(f"Vector store created and saved to {vectorstore_path}.")

        retriever = vectorstore.as_retriever()

        # 2. Create a retrieval chain that can be wrapped into a tool
        retrieval_prompt = ChatPromptTemplate.from_template("""Answer the user's question based ONLY on the following context:
        <context>
        {context}
        </context>

        Question: {input}""")
        
        document_chain = create_stuff_documents_chain(self.llm, retrieval_prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        @tool
        def query_exploit_documentation(query: str) -> str:
            """
            Searches the local exploit documentation (cloned from a git repo) to answer questions about how to set up, configure, or run an exploit.
            Use this tool whenever you need to understand the steps described in the repository's README or source code.
            """
            logger.info(f"Querying documentation with: '{query}'")
            response = retrieval_chain.invoke({"input": query})
            return response['answer']

        # ...
        # 现在我们调用的是一个真正的 Python 函数，不会再有错误了
        bound_shell_tool = lambda command: _run_shell_command_logic(command=command, path=self.doc_dir)
        # 从原始逻辑函数复制文档字符串，以便 LLM 理解
        bound_shell_tool.__doc__ = _run_shell_command_logic.__doc__ 

        # 将我们绑定好参数的 lambda 函数包装成一个 LLM 可以使用的工具
        shell_command_tool = tool("run_shell_command")(bound_shell_tool)

        return [query_exploit_documentation, shell_command_tool]
        
    def _create_agent_executor(self):
        """Builds the agent executor using LangChain Expression Language (LCEL)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert cybersecurity penetration tester. Your goal is to execute a plan to exploit a target system. You have access to tools to run shell commands and query local documentation. Think step-by-step. First, analyze the situation, then decide on an action (use a tool or respond to the user). Continue until the task is complete or you determine it's not possible."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

        # Wrap it in a runnable that manages history
        agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return agent_with_history

    def get_session_history(self, session_id: str):
        """Callback function to retrieve message history for a given session."""
        if session_id not in self.memory_store:
            # Using FileChatMessageHistory to persist conversations
            history_file = os.path.join(self.doc_dir, f".chat_history_{session_id}.json")
            self.memory_store[session_id] = FileChatMessageHistory(history_file)
        return self.memory_store[session_id]

    def invoke(self, message: str, session_id: str = "default"):
        """Sends a message to the agent and gets the response."""
        config = {"configurable": {"session_id": session_id}}
        # The agent executor now handles the entire loop of thinking,
        # using tools, and responding.
        response = self.agent_executor.invoke({"input": message}, config=config)
        return response['output']

def main():
    # --- Setup ---
    start_time = time.time()
    
    repo_url = execution_config.get('repo_url')
    if repo_url:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        doc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), execution_config['doc_dir'], repo_name))
        os.makedirs(doc_dir, exist_ok=True)
        if not os.path.exists(os.path.join(doc_dir, '.git')):
            logger.info(f"Cloning repository {repo_url} into {doc_dir}")
            Repo.clone_from(repo_url, doc_dir)
        else:
            logger.info(f"Repository already exists at {doc_dir}. Skipping clone.")
    else:
        doc_dir = os.path.join(os.path.dirname(__file__), execution_config['doc_dir'])
    
    logger.info(f"Using documentation and execution directory: {doc_dir}\n")
    
    # --- Agent Initialization ---
    exec_agent = ExecutionAgent(doc_dir=doc_dir)
    
    # --- Execution Flow ---
    # 1. Initial analysis using the agent's RAG tool
    msg1 = PentestAgentPrompt.execution_init_exploit_analysis
    res1 = exec_agent.invoke(msg1)
    logger.info(f"Initial analysis response:\n{res1}")
    
    # 2. (Optional) Interact with ReconAgent
    final_guide_prompt = ""
    recon_res = None
    if execution_config.get('current_topic'):
        recon_agent = ReconAgent()
        query_msg = f"Based on the known information, try to provide the information needed listed here:\n{res1}"
        recon_res = recon_agent.run(topic=execution_config['current_topic'], user_input=query_msg)
        if recon_res:
            logger.info(f"Reconnaissance agent response:\n{recon_res}")
        else:
            logger.warning("No response from recon agent.")

    # 3. Formulate the final execution prompt and let the agent take over
    target_ip = execution_config['target_ip']
    target_port = execution_config['target_port']
    attacker_ip = execution_config['attacker_ip']
    
    addn_msg = (
        f"\nThe victim application is running on {target_ip}:{target_port}."
        f"\nThe attacker machine IP is {attacker_ip}."
    )
    if recon_res:
        final_guide_prompt = f"Here are the answers to your questions from a reconnaissance agent: {recon_res}\n{addn_msg}\n{PentestAgentPrompt.execution_step_by_step_guide}"
    else:
        final_guide_prompt = f"{addn_msg}\n{PentestAgentPrompt.execution_step_by_step_guide}"

    logger.info("--- Starting automated execution phase ---")
    res2 = exec_agent.invoke(final_guide_prompt)
    logger.info(f"Execution phase complete. Final response:\n{res2}")
    
    prepare_end_time = time.time() # Mark end of main interaction
    
    # 4. Final summary
    summary_res = exec_agent.invoke(PentestAgentPrompt.execution_summary)
    logger.info(f"Execution summary:\n{summary_res}")
    
    # --- Timing and conclusion ---
    execution_end_time = time.time()
    prepare_time = prepare_end_time - start_time
    exec_time = execution_end_time - prepare_end_time
    total_time = execution_end_time - start_time

    logger.info(f"Preparation Time: {prepare_time:.2f} seconds")
    logger.info(f"Execution/Summary Time: {exec_time:.2f} seconds")
    logger.info(f"Total Agent Run Time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()