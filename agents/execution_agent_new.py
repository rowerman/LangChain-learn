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
        
        # 从配置文件读取超时配置
        try:
            timeout = config.get('models', {}).get('openai', {}).get('timeout', 180)
        except NameError:
            # 如果 config 未定义，使用默认值
            timeout = 180
        
        # 配置 LLM，添加超时和重试
        # 注意：对于 agent executor，我们使用非流式模式，因为流式输出在 agent 中难以处理
        llm = ChatOpenAI(
            model=model_name, 
            temperature=0, 
            streaming=False,  # Agent executor 不支持流式输出
            max_retries=3,
            request_timeout=timeout
        )
        logger.info(f"LLM configured with timeout={timeout}s, max_retries=3")
        return llm

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
        
        # 为文档查询创建一个单独的 LLM 实例，支持流式输出以显示进度
        try:
            timeout = config.get('models', {}).get('openai', {}).get('timeout', 180)
        except NameError:
            timeout = 180
        
        # 创建一个用于文档查询的 LLM（可以支持流式输出）
        doc_llm = ChatOpenAI(
            model=execution_config['model'],
            temperature=0,
            streaming=False,  # 暂时关闭流式输出，因为 retrieval chain 可能不支持
            max_retries=3,
            request_timeout=timeout
        )
        
        document_chain = create_stuff_documents_chain(doc_llm, retrieval_prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        @tool
        def query_exploit_documentation(query: str) -> str:
            """
            Searches the local exploit documentation (cloned from a git repo) to answer questions about how to set up, configure, or run an exploit.
            Use this tool whenever you need to understand the steps described in the repository's README or source code.
            """
            logger.info(f"Querying documentation with: '{query}'")
            try:
                import time
                start_time = time.time()
                logger.info("Starting RAG retrieval...")
                
                response = retrieval_chain.invoke({"input": query})
                
                elapsed_time = time.time() - start_time
                answer = response.get('answer', 'No answer found')
                
                logger.info(f"RAG retrieval completed in {elapsed_time:.2f}s")
                logger.info(f"Retrieved answer length: {len(answer)} characters")
                logger.info(f"Answer preview: {answer[:200]}..." if len(answer) > 200 else f"Answer: {answer}")
                
                return answer
            except Exception as e:
                logger.error(f"Error querying documentation: {e}")
                logger.exception("Full traceback:")
                return f"Error querying documentation: {str(e)}"

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
            ("system", "You are an expert cybersecurity penetration tester. "
            "Your goal is to execute a plan to exploit a target system. "
            "You have access to tools to run shell commands and query local documentation. "
            "Think step-by-step. First, analyze the situation, then decide on an action (use a tool or respond to the user). "
            "Continue until the task is complete or you determine it's not possible."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        # 设置 verbose=True 以显示 agent 的执行步骤
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,
            max_iterations=15,  # 限制最大迭代次数，防止无限循环
            max_execution_time=300,  # 设置最大执行时间（5分钟）
            handle_parsing_errors=True  # 处理解析错误
        )

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
        import time
        from functools import wraps
        
        config = {"configurable": {"session_id": session_id}}
        
        logger.info(f"Invoking agent with message (length: {len(message)} chars)")
        logger.debug(f"Message content: {message[:500]}..." if len(message) > 500 else f"Message: {message}")
        
        start_time = time.time()
        
        try:
            # The agent executor now handles the entire loop of thinking,
            # using tools, and responding.
            logger.info("Agent executor starting...")
            response = self.agent_executor.invoke({"input": message}, config=config)
            
            elapsed_time = time.time() - start_time
            output = response.get('output', '')
            
            logger.info(f"Agent executor completed in {elapsed_time:.2f}s")
            logger.info(f"Response length: {len(output)} characters")
            
            # 打印响应的预览
            if output:
                preview = output[:500] + "..." if len(output) > 500 else output
                logger.info(f"Response preview: {preview}")
            else:
                logger.warning("Agent returned empty response")
            
            return output
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Agent invocation failed after {elapsed_time:.2f}s: {e}")
            logger.exception("Full traceback:")
            raise

def extract_repo_info(url: str) -> tuple:
    """
    从 GitHub URL 中提取仓库信息和路径。
    
    支持的格式：
    - https://github.com/user/repo/blob/master/path/to/dir
    - https://github.com/user/repo/tree/master/path/to/dir
    - https://github.com/user/repo
    - https://github.com/user/repo.git
    
    返回：(repo_url, subdirectory_path, branch)
    """
    if not url:
        return None, None, None
    
    # 移除末尾的斜杠
    url = url.rstrip('/')
    
    # 如果是 SSH URL，转换为 HTTPS
    if url.startswith('git@github.com:'):
        url = url.replace('git@github.com:', 'https://github.com/')
    
    branch = 'master'  # 默认分支
    subdirectory = None
    
    # 处理 /blob/ 路径（文件或目录）
    if '/blob/' in url:
        parts = url.split('/blob/')
        repo_base = parts[0]
        path_parts = parts[1].split('/', 1)
        branch = path_parts[0]
        if len(path_parts) > 1:
            subdirectory = path_parts[1]
    
    # 处理 /tree/ 路径（目录）
    elif '/tree/' in url:
        parts = url.split('/tree/')
        repo_base = parts[0]
        path_parts = parts[1].split('/', 1)
        branch = path_parts[0]
        if len(path_parts) > 1:
            subdirectory = path_parts[1]
    
    else:
        repo_base = url
    
    # 确保 URL 以 .git 结尾（Git 克隆需要）
    if not repo_base.endswith('.git'):
        repo_url = repo_base + '.git'
    else:
        repo_url = repo_base
    
    return repo_url, subdirectory, branch

def clone_with_sparse_checkout(repo_url: str, target_dir: str, subdirectory: str = None, branch: str = 'master', max_retries: int = 3):
    """
    使用 sparse checkout 克隆仓库的特定目录。
    
    Args:
        repo_url: Git 仓库 URL
        target_dir: 目标目录
        subdirectory: 要检出的子目录路径（相对于仓库根目录）
        branch: 要检出的分支名
        max_retries: 最大重试次数
    """
    import tempfile
    import shutil
    
    # 创建临时目录用于初始克隆
    temp_dir = tempfile.mkdtemp(prefix='git_sparse_')
    
    def run_git_command(cmd, cwd, description, timeout=180, show_output=False):
        """运行 Git 命令并显示进度"""
        logger.info(f"Running: {description}")
        for attempt in range(max_retries):
            try:
                if show_output:
                    # 显示实时输出
                    process = subprocess.Popen(
                        cmd, 
                        cwd=cwd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    # 实时打印输出
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            logger.debug(f"Git: {line}")
                    process.wait()
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, cmd)
                else:
                    result = subprocess.run(
                        cmd, 
                        cwd=cwd, 
                        check=True, 
                        capture_output=True, 
                        text=True,
                        timeout=timeout
                    )
                    if result.stdout:
                        logger.debug(f"Git output: {result.stdout[:200]}")
                logger.info(f"✓ {description} completed")
                return True
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout during {description} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # 指数退避
                    continue
                raise
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else (e.stdout if hasattr(e, 'stdout') else str(e))
                logger.warning(f"Error during {description} (attempt {attempt + 1}/{max_retries}): {error_msg[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # 指数退避
                    continue
                raise
            except Exception as e:
                logger.warning(f"Unexpected error during {description} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        return False
    
    try:
        logger.info(f"Initializing sparse checkout for {repo_url}")
        logger.info(f"Target directory: {target_dir}")
        if subdirectory:
            logger.info(f"Subdirectory to checkout: {subdirectory}")
        logger.info(f"Branch: {branch}")
        
        # 1. 创建目标目录
        os.makedirs(target_dir, exist_ok=True)
        
        # 2. 初始化 Git 仓库
        run_git_command(['git', 'init'], target_dir, "Initializing Git repository")
        
        # 3. 添加 remote
        run_git_command(['git', 'remote', 'add', 'origin', repo_url], target_dir, "Adding remote origin")
        
        # 4. 启用 sparse checkout
        run_git_command(['git', 'config', 'core.sparseCheckout', 'true'], target_dir, "Enabling sparse checkout")
        
        # 5. 配置 Git 超时和缓冲区大小（处理大仓库）
        run_git_command(['git', 'config', 'http.postBuffer', '524288000'], target_dir, "Setting Git buffer size")
        run_git_command(['git', 'config', 'http.lowSpeedLimit', '0'], target_dir, "Configuring Git low speed limit")
        run_git_command(['git', 'config', 'http.lowSpeedTime', '0'], target_dir, "Configuring Git low speed time")
        
        # 6. 配置 sparse checkout 路径
        sparse_checkout_file = os.path.join(target_dir, '.git', 'info', 'sparse-checkout')
        os.makedirs(os.path.dirname(sparse_checkout_file), exist_ok=True)
        
        with open(sparse_checkout_file, 'w') as f:
            if subdirectory:
                # 确保路径以 / 开头，并且不以 / 结尾（除非是根目录）
                sparse_path = subdirectory.strip('/')
                if sparse_path:
                    f.write(f"{sparse_path}\n")
                    f.write(f"{sparse_path}/*\n")  # 包含子目录
                else:
                    f.write("/*\n")
            else:
                f.write("/*\n")
        
        # 7. 获取远程分支信息（显示进度）
        try:
            run_git_command(
                ['git', 'fetch', 'origin', branch], 
                target_dir, 
                f"Fetching branch '{branch}'", 
                timeout=300,
                show_output=True
            )
        except subprocess.CalledProcessError:
            # 如果分支不存在，尝试 main
            if branch == 'master':
                logger.warning(f"Branch 'master' not found, trying 'main'")
                branch = 'main'
                run_git_command(
                    ['git', 'fetch', 'origin', branch], 
                    target_dir, 
                    f"Fetching branch '{branch}'", 
                    timeout=300,
                    show_output=True
                )
            else:
                raise
        
        # 8. 检出指定分支
        run_git_command(['git', 'checkout', branch], target_dir, f"Checking out branch '{branch}'")
        
        logger.info(f"✓ Successfully checked out {subdirectory if subdirectory else 'all files'} from branch {branch}")
        
    except Exception as e:
        logger.error(f"Failed to perform sparse checkout: {e}")
        logger.exception("Full traceback:")
        # 清理失败的目录
        if os.path.exists(target_dir):
            logger.info(f"Cleaning up failed directory: {target_dir}")
            shutil.rmtree(target_dir, ignore_errors=True)
        raise
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    # --- Setup ---
    start_time = time.time()
    
    repo_url = execution_config.get('repo_url')
    if repo_url:
        # 从 URL 中提取仓库信息和路径
        normalized_url, subdirectory, branch = extract_repo_info(repo_url)
        logger.info(f"Original URL: {repo_url}")
        logger.info(f"Repository URL: {normalized_url}")
        if subdirectory:
            logger.info(f"Subdirectory: {subdirectory}")
        logger.info(f"Branch: {branch}")
        
        # 从规范化后的 URL 提取仓库名
        repo_name = normalized_url.split('/')[-1].replace('.git', '')
        # 如果指定了子目录，将其包含在目录名中
        if subdirectory:
            # 清理子目录路径，用于创建目录名
            safe_subdir = subdirectory.replace('/', '_').replace('\\', '_')
            repo_name = f"{repo_name}_{safe_subdir}"
        
        doc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), execution_config['doc_dir'], repo_name))
        os.makedirs(doc_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(doc_dir, '.git')):
            if subdirectory:
                # 使用 sparse checkout 只克隆指定目录
                logger.info(f"Cloning repository with sparse checkout (only {subdirectory})")
                try:
                    clone_with_sparse_checkout(normalized_url, doc_dir, subdirectory, branch)
                    logger.info(f"Successfully cloned {subdirectory} from repository into {doc_dir}")
                except Exception as e:
                    logger.error(f"Failed to clone repository with sparse checkout: {e}")
                    logger.error(f"Falling back to full clone...")
                    # 如果 sparse checkout 失败，回退到完整克隆（使用浅克隆）
                    logger.info("Attempting shallow clone as fallback...")
                    try:
                        # 使用浅克隆（depth=1）只克隆最新提交，减少下载量
                        import shutil
                        if os.path.exists(doc_dir):
                            shutil.rmtree(doc_dir, ignore_errors=True)
                        
                        # 使用 subprocess 以便显示进度
                        logger.info("Starting shallow clone (depth=1)...")
                        process = subprocess.Popen(
                            ['git', 'clone', '--depth', '1', '--branch', branch, normalized_url, doc_dir],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1
                        )
                        
                        # 实时显示克隆进度
                        for line in process.stdout:
                            line = line.strip()
                            if line:
                                logger.info(f"Git clone: {line}")
                        
                        process.wait()
                        if process.returncode != 0:
                            raise subprocess.CalledProcessError(process.returncode, 'git clone')
                        
                        logger.info(f"✓ Successfully cloned repository (shallow) into {doc_dir}")
                    except Exception as fallback_error:
                        logger.error(f"Full clone also failed: {fallback_error}")
                        logger.exception("Full traceback:")
                        raise
            else:
                # 没有指定子目录，使用浅克隆（只克隆最新提交）
                logger.info(f"Cloning repository (shallow clone) into {doc_dir}")
                try:
                    Repo.clone_from(normalized_url, doc_dir, depth=1)
                    logger.info(f"Successfully cloned repository into {doc_dir}")
                except Exception as e:
                    logger.error(f"Failed to clone repository: {e}")
                    logger.error(f"Please check if the URL is correct: {normalized_url}")
                    raise
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