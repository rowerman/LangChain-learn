import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import subprocess
import logging
import time
import yaml
import dotenv
from pydantic import BaseModel, Field
# --- 核心 LangChain 1.0 导入 ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
# langchain_openai 是一个例子，你可以换成 langchain_anthropic, langchain_google_vertexai 等
from langchain_openai import ChatOpenAI 
from langchain_community.chat_message_histories import FileChatMessageHistory

# --- 假设的本地工具 (保持不变) ---
# 假设这些文件存在于你的项目中
# utils/prompt.py
# utils/model_manager.py

# 这是一个模拟，因为我们没有你的原始文件
class PentestAgentPrompt:
    recon_init = "Start reconnaissance on target <Target-Ip>. Provide your analysis, next step, and an executable command in JSON format."
    recon_summary = "Please provide a final summary of your findings."

# 这是一个模拟，用来替代 get_model。在 LangChain 1.0 中，我们通常直接实例化模型。
def get_model_v2(model_name: str):
    """
    LangChain 1.0 style model loader.
    Replace this with your actual model provider, e.g., ChatOllama, ChatAnthropic, etc.
    """
    # 示例：使用 OpenAI。确保你的 .env 文件中有 OPENAI_API_KEY
    if "gpt" in model_name:
        # LangChain 会自动从环境变量加载 API 密钥
        # 增加了 max_retries 来替代旧代码中的手动重试循环
        return ChatOpenAI(model=model_name, temperature=0, max_retries=3, request_timeout=30)
    # 在这里添加对其他模型的支持，例如 Ollama
    # from langchain_community.chat_models import ChatOllama
    # if "ollama" in model_name:
    #     return ChatOllama(model=model_name)
    raise ValueError(f"Model {model_name} not supported in this example.")


# --- 日志和配置加载 (保持不变) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('recon_agent_v2.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

dotenv.load_dotenv()
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
logger.info(f"Loading config from: {config_path}")
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded successfully")
except Exception as e:
    logger.error(f"Failed to load config: {str(e)}")
    sys.exit(1)

recon_config = config['runtime']['recon']
model_name = recon_config['model']
logger.info(f"Recon config: {json.dumps(recon_config, indent=2)}")

# --- Pydantic 模型 (保持不变, 但现在更重要) ---
class ReconResponse(BaseModel):
    analysis: str = Field(description="Analysis of the previous step")
    next_step: str = Field(description="What to do next")
    executable: str = Field(description="Command to execute, or 'None' if no command is needed")

# --- 重构后的 ReconAgent ---
class ReconAgent:
    def __init__(self):
        logger.info("Initializing ReconAgent (v2 with LCEL)")
        self.memory_dir = recon_config['memory_dir']
        os.makedirs(self.memory_dir, exist_ok=True)
        
        try:
            # 直接实例化模型，这是 LangChain 1.0 的推荐做法
            self.llm = get_model_v2(model_name)
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise  # 在初始化失败时抛出异常

        # 1. 定义提示模板，包含历史消息占位符
        # 我们指导模型以 JSON 格式输出，并提供 Pydantic 模型的 schema
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful penetration testing assistant. "
                       "Analyze the user's request and the results of previous commands. "
                       "When performing an nmap scan, you MUST use the `-p-` flag to scan all 65535 ports. For example: `nmap -p- -sS -sV <target_ip>`. "
                       "Your response MUST be a JSON object that conforms to the provided schema. "
                       "Do not add any other text or markdown formatting around the JSON."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # 2. 创建核心链 (LCEL): prompt | model | structured_output_parser
        # .with_structured_output 会自动处理提示、解析和重试，以确保得到 Pydantic 对象
        chain = prompt | self.llm.with_structured_output(ReconResponse)

        # 3. 使用 RunnableWithMessageHistory 包装链，以自动管理历史记录
        # 这将替代所有手动的内存管理方法
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_chat_history,  # 提供一个函数来根据 session_id 获取历史记录对象
            input_messages_key="input",
            history_messages_key="history",
        )
        logger.info("Agent chain with message history initialized.")

    def _get_chat_history(self, session_id: str) -> FileChatMessageHistory:
        """
        根据 session_id (即 topic) 获取或创建文件聊天历史记录。
        这替代了旧的 get_memory, load_memory, save_memory 方法。
        """
        memory_file = os.path.join(self.memory_dir, f"{session_id}.json")
        return FileChatMessageHistory(memory_file)

    def run(self, topic: str, user_input: str) -> ReconResponse:
        """
        使用 LCEL 链运行一个回合的对话。
        这个方法替代了旧的 send_message 和 run_thread。
        """
        logger.info(f"Running chain for topic '{topic}'")
        
        # 'configurable' 字典是 LangChain 1.0 的标准方式，用于传递运行时参数，如 session_id
        config = {"configurable": {"session_id": topic}}
        
        try:
            # 调用链。RunnableWithMessageHistory 会自动加载历史，
            # 将新消息添加到历史中，调用链，然后保存结果。
            response = self.chain_with_history.invoke({"input": user_input}, config=config)
            return response
        except Exception as e:
            logger.error(f"Error invoking chain for topic {topic}: {e}")
            # 返回一个表示错误的 Pydantic 对象，使主循环可以优雅地处理它
            return ReconResponse(
                analysis=f"An error occurred: {e}",
                next_step="Review the error and try again.",
                executable="None"
            )

    def run_shell_command(self, command: str) -> str:
        """运行 shell 命令 (保持不变)"""
        logger.info(f"Executing command: {command}")
        try:
            result = subprocess.run(
                command, shell=True, check=True, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                text=True, timeout=600 # 增加了超时时间以适应全端口扫描
            )
            output = result.stdout
            if not output.strip():
                return "Command executed successfully with no output."
            logger.info(f"Command executed successfully. Output length: {len(output)}")
            return output
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with error:\n{e.stderr}"
            logger.error(error_msg)
            return error_msg
        except subprocess.TimeoutExpired:
            # 同样增加了超时时间
            error_msg = "Command timed out after 600 seconds"
            logger.error(error_msg)
            return error_msg


def main():
    logger.info("Starting ReconAgent (v2)")
    start_time = time.time()
    
    try:
        recon_agent = ReconAgent()
    except Exception as e:
        logger.critical(f"Failed to initialize ReconAgent: {str(e)}")
        return
    
    curr_topic = config['runtime']['recon'].get('current_topic', 'default_topic')
    target_ip = config['runtime']['recon'].get('target_ip', 'unknown_ip')
    
    logger.info(f"Current topic: {curr_topic}")
    logger.info(f"Target IP: {target_ip}")
    
    # 准备初始输入
    recon_init_prompt = PentestAgentPrompt().recon_init.replace("<Target-Ip>", target_ip)
    # 组合成一个更清晰的初始任务
    initial_input = (
        f"{recon_init_prompt}\n"
        f"My ultimate goal is to exploit the target host at {target_ip}. Begin the reconnaissance phase."
    )
    logger.info(f"Initial input: {initial_input[:150]}...")

    max_steps = 10
    current_input = initial_input

    for step in range(max_steps):
        logger.info(f"--- Step {step + 1}/{max_steps} ---")
        
        # 单一调用来处理对话
        response_obj = recon_agent.run(curr_topic, current_input)
        
        # 现在我们直接得到一个 Pydantic 对象，不再需要解析 JSON
        print("\n==============================")
        print(f"[LLM Analysis]\n{response_obj.analysis}")
        print(f"[Next Step]\n{response_obj.next_step}")
        print(f"[Executable Command]\n{response_obj.executable}")
        print("==============================\n")

        cmd = response_obj.executable
        if cmd and cmd.lower() != 'none':
            cmd_res = recon_agent.run_shell_command(cmd)
            print(f"[Command Execution Result]\n{cmd_res[:1000]}...") # 打印部分结果以防过长
            
            # 准备下一步的输入，将命令结果反馈给 LLM
            current_input = (
                "The command was executed. Here is the output. "
                "Please analyze it and decide the next step.\n\n"
                f"---COMMAND OUTPUT---\n{cmd_res}"
            )
        else:
            logger.info("No more commands to execute. Proceeding to summary.")
            break
    else:
        logger.warning(f"Reached max steps ({max_steps}). Forcing summary.")

    logger.info("Requesting final summary")
    summary_input = PentestAgentPrompt.recon_summary
    final_response = recon_agent.run(curr_topic, summary_input)
    
    print("\n--- FINAL SUMMARY ---")
    print(final_response.analysis)
    print("---------------------\n")

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Reconnaissance agent execution completed in {execution_time:.2f} seconds")
    print(f"Reconnaissance agent execution completed in {execution_time:.2f} seconds")
    # 无需手动保存内存，FileChatMessageHistory 已在每一步自动完成

if __name__ == "__main__":
    main()