import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import logging
import subprocess
import yaml
from typing import List, Dict

import dotenv
from pydantic import BaseModel, Field

# --- LangChain 1.0 Core Imports ---
from langchain_core.prompts import ChatPromptTemplate
# Example with OpenAI, can be swapped with other providers
from langchain_openai import ChatOpenAI

# --- Configuration and Logging Setup (Improved) ---
# Use a more robust logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("planning_agent_v2.log"),
        logging.StreamHandler(sys.stdout) # Log to both file and console
    ]
)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

try:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    planning_config = config['runtime']['planning']
    cvemap_config = planning_config['cvemap']
    logger.info("Configuration loaded successfully.")
except FileNotFoundError:
    logger.error(f"Configuration file not found at {config_path}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    sys.exit(1)

# --- Local Utility Imports (assumed to be correct) ---
# It's good practice to wrap local imports in a try-except if they are complex
# 暂时注释掉 try...except
# try:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.merge_scores import merge
from utils.version_limit import get_affected_cve
# The original code had a conditional import, which is a good pattern
if config['runtime']['planning'].get('economic_mode', False):
    from utils.cve_info_ec import get_exp_info
else:
    from utils.cve_info import get_exp_info
# except ImportError as e:
#     print(f"Error importing local utils: {e}. Make sure your PYTHONPATH is set correctly.")
#     sys.exit(1)


# --- LangChain 1.0 Style Model Loader ---
def get_llm_model(model_name: str):
    """Initializes and returns a LangChain 1.0 model."""
    logger.info(f"Initializing model: {model_name}")
    # This example uses OpenAI. You can easily add logic for other providers
    # like Ollama, Anthropic, etc.
    if "gpt" in model_name:
        # LangChain automatically uses OPENAI_API_KEY from environment variables
        return ChatOpenAI(model=model_name, temperature=0, max_retries=2)
    # Example for Ollama
    # from langchain_community.chat_models import ChatOllama
    # if "ollama" in model_name:
    #     return ChatOllama(model=model_name)
    raise ValueError(f"Model '{model_name}' is not supported in this configuration.")


# --- Pydantic model for structured LLM output ---
class ExploitPlanSummary(BaseModel):
    """Defines the structure for the exploitation plan summary."""
    summary: str = Field(description="A brief summary of the most promising vulnerabilities and exploits found.")
    top_cve: str = Field(description="The single most critical CVE ID to investigate first, e.g., 'CVE-2023-12345'. If none, state 'N/A'.")
    recommended_action: str = Field(description="The next concrete step a penetration tester should take.")


# +++ 请使用这个带有调试功能的版本替换 cvemap_product 函数 +++

# +++ 请使用这个最终修正版的函数完整替换旧函数 +++

def cvemap_product(product: str, output_dir: str, cvemap_cfg: Dict) -> List[Dict]:
    """使用 vulnx 工具为指定产品查找 CVE。此版本能正确处理 vulnx 返回的复杂 JSON 对象。"""
    os.makedirs(output_dir, exist_ok=True)
    vulnx_json_path = os.path.join(output_dir, "vulnx.json")
    lower_product = product.lower()
    
    all_results = []
    # 恢复默认 limit 或根据您的需求调整
    limit = cvemap_cfg.get("limit", 10) 
    offset = 0
    max_entries = cvemap_cfg.get("max_entry")
    min_year = cvemap_cfg.get("min_year")
    max_year = cvemap_cfg.get("max_year")
    # 添加最大 offset 限制，防止无限循环
    max_offset = 10000  # 设置一个合理的上限

    logger.info(f"Starting vulnx search for product '{lower_product}'...")

    while True:
        if max_entries and len(all_results) >= max_entries:
            break
        
        # 检查 offset 是否超过最大限制
        if offset >= max_offset:
            logger.warning(f"Reached maximum offset limit ({max_offset}). Stopping search.")
            break

        # 使用产品名作为位置参数，而不是 -p 参数
        vulnx_command = [
            "vulnx", "search", lower_product,
            "-n", str(limit),
            "--offset", str(offset),
            "--detailed",
            "-j"
        ]
        
        logger.info(f"Executing vulnx command: {' '.join(vulnx_command)}")

        try:
            result = subprocess.run(
                vulnx_command, check=True, capture_output=True, text=True, timeout=120
            )
            if not result.stdout.strip():
                logger.info("vulnx returned no more results. Ending search.")
                break
            
            # --- 核心修正点在这里 ---
            # 1. 解析完整的 JSON 对象
            decoded_json = json.loads(result.stdout)
            
            # 2. 从对象中提取 'results' 键对应的列表
            current_batch = decoded_json.get('results', []) 

            if not current_batch:
                logger.info("vulnx returned a response with an empty 'results' list. Ending search.")
                break

            # 记录本次获取到的原始结果数量
            batch_size_before_filter = len(current_batch)

            # 3. 遍历这个包含 CVE 详细信息的字典列表
            for cve_item in current_batch:
                # 确保 cve_item 是字典并且包含 'cve_id'
                if not isinstance(cve_item, dict) or 'cve_id' not in cve_item:
                    continue

                cve_id_str = cve_item['cve_id']
                
                # 年份过滤逻辑现在可以正常工作了
                try:
                    year = int(cve_id_str.split('-')[1])
                except (IndexError, ValueError):
                    continue
                
                if (max_year and year > max_year) or (min_year and year < min_year):
                    continue
                
                # 将过滤后的 CVE ID 包装成下游代码需要的字典格式
                all_results.append({'cve_id': cve_id_str})

                if max_entries and len(all_results) >= max_entries:
                    break
            
            # 改进的终止条件：
            # 1. 如果返回的结果数量小于 limit，说明已经到达末尾
            # 2. 如果已经达到最大条目数，停止
            # 3. 如果 offset > 0 且返回结果为空，说明已经到达末尾
            if batch_size_before_filter < limit:
                logger.info(f"Received {batch_size_before_filter} results (less than limit {limit}). No more pages available.")
                break
            
            if max_entries and len(all_results) >= max_entries:
                break
                
            offset += limit

        except subprocess.CalledProcessError as e:
            logger.error(f"vulnx command failed with exit code {e.returncode}.")
            logger.error(f"Command executed: {' '.join(vulnx_command)}")
            logger.error(f"STDERR: {e.stderr}")
            break 
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from vulnx at offset {offset}.")
            logger.warning(f"Raw output: {result.stdout}")
            break
        except FileNotFoundError:
            logger.error("The 'vulnx' command was not found. Is it installed and in your PATH?")
            raise

    if max_entries:
        all_results = all_results[:max_entries]

    with open(vulnx_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"vulnx search complete. Found {len(all_results)} potential CVEs.")
    return all_results


# --- Main Application Logic ---
def main():
    # 1. Initialize Model
    model_name = planning_config['model']
    try:
        llm = get_llm_model(model_name)
    except (ValueError, Exception) as e:
        logger.critical(f"Failed to initialize LLM: {e}")
        return

    # 2. Setup environment from config
    keyword = planning_config['keyword']
    app = planning_config['app']
    version = planning_config.get('version') # Use .get for optional keys
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), planning_config['output_dir'])
    res_dir = os.path.join(output_dir, keyword)
    os.makedirs(res_dir, exist_ok=True)
    
    logger.info(f"Starting planning agent for App: '{app}', Version: '{version or 'any'}'")

    # 3. Check if a specific CVE is specified
    specific_cve = cvemap_config.get('specific_cve')
    
    if specific_cve:
        # 如果指定了具体的 CVE，直接使用该 CVE，跳过 vulnx 搜索
        logger.info(f"Specific CVE specified: {specific_cve}. Skipping vulnx search.")
        
        # 验证 CVE 格式
        if not specific_cve.upper().startswith('CVE-'):
            logger.warning(f"Invalid CVE format: {specific_cve}. Expected format: CVE-YYYY-NNNNN")
            return
        
        # 应用年份过滤（如果配置了）
        min_year = cvemap_config.get('min_year')
        max_year = cvemap_config.get('max_year')
        
        try:
            cve_year = int(specific_cve.split('-')[1])
            if (max_year and cve_year > max_year) or (min_year and cve_year < min_year):
                logger.warning(f"CVE {specific_cve} (year: {cve_year}) is outside the configured year range "
                             f"(min_year: {min_year}, max_year: {max_year}). Exiting.")
                return
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to extract year from CVE {specific_cve}: {e}")
            return
        
        # 如果指定了版本，需要先获取 CVE 描述信息以进行版本过滤
        cve_ids = [specific_cve.upper()]
        
        # 为了进行版本过滤，需要先获取 CVE 描述信息
        if version:
            logger.info(f"Version specified ({version}). Fetching CVE description for version filtering...")
            # 导入 cvemap_search 函数以获取 CVE 描述
            from utils.cve_info import cvemap_search
            cvemap_res_dir = os.path.join(res_dir, "CVEMAP")
            os.makedirs(cvemap_res_dir, exist_ok=True)
            info_dir = os.path.join(cvemap_res_dir, "info")
            os.makedirs(info_dir, exist_ok=True)
            
            # 获取 CVE 详细信息
            cvemap_json = cvemap_search(specific_cve.upper(), info_dir)
            if cvemap_json is None:
                logger.warning(f"Failed to fetch CVE description for {specific_cve}. Skipping version filter.")
                cvemap_results = [{'cve_id': specific_cve.upper()}]
            else:
                # 提取 CVE 描述信息
                results_list = cvemap_json.get('results', [])
                if results_list and len(results_list) > 0:
                    cve_details = results_list[0]
                    cve_description = cve_details.get('description', 'No description available.')
                    cvemap_results = [{
                        'cve_id': specific_cve.upper(),
                        'cve_description': cve_description
                    }]
                    logger.info(f"Fetched CVE description for version filtering.")
                else:
                    logger.warning(f"Could not extract CVE description. Skipping version filter.")
                    cvemap_results = [{'cve_id': specific_cve.upper()}]
        else:
            # 如果没有指定版本，直接使用 CVE ID
            cvemap_results = [{'cve_id': specific_cve.upper()}]
            logger.info(f"Using specific CVE: {cve_ids[0]} (no version filter)")
        
        # 保存到 CVEMAP 目录以保持一致性
        cvemap_res_dir = os.path.join(res_dir, "CVEMAP")
        os.makedirs(cvemap_res_dir, exist_ok=True)
        vulnx_json_path = os.path.join(cvemap_res_dir, "vulnx.json")
        with open(vulnx_json_path, "w", encoding="utf-8") as f:
            json.dump(cvemap_results, f, indent=2)
        
        # 如果指定了版本，进行版本过滤
        if version and 'cve_description' in cvemap_results[0]:
            logger.info(f"Filtering specific CVE for version '{version}'. Note: This may not be perfectly accurate.")
            limited_lst = get_affected_cve(cvemap_results, version)
            if limited_lst:
                cve_ids = [item['cve_id'] for item in limited_lst]
                logger.info(f"Specific CVE {specific_cve} affects version {version}. Proceeding with analysis.")
            else:
                logger.warning(f"Specific CVE {specific_cve} does not appear to affect version {version} based on description analysis.")
                logger.info(f"However, since a specific CVE was explicitly requested, proceeding with analysis anyway.")
                # 继续处理，因为用户明确指定了该 CVE
        elif version:
            logger.info(f"Version specified but CVE description unavailable. Proceeding with specific CVE analysis.")
    else:
        # 原有的逻辑：使用 vulnx 搜索 CVE
        cvemap_res_dir = os.path.join(res_dir, "CVEMAP")
        cvemap_results = cvemap_product(app.lower().replace(' ', '_'), cvemap_res_dir, cvemap_config)
        
        # 4. Filter data based on logic
        if version:
            logger.info(f"Filtering CVEs for version '{version}'. Note: This may not be perfectly accurate.")
            limited_lst = get_affected_cve(cvemap_results, version)
            cve_ids = [item['cve_id'] for item in limited_lst]
            logger.info(f"Found {len(cve_ids)} CVEs potentially affecting version {version}: {cve_ids}")
        else:
            logger.info("No version constraint set. Using all found CVEs.")
            cve_ids = [item["cve_id"] for item in cvemap_results]

        if not cve_ids:
            logger.warning("No relevant CVEs found after filtering. Exiting.")
            return

    # 5. Process data with more tools/scripts
    exploit_searching_time, exploit_analysis_time = get_exp_info(cve_ids, res_dir, app)
    
    plan_filename = "plan_ec.json" if planning_config.get('economic_mode') else "plan.json"
    final_plan_path = os.path.join(res_dir, plan_filename)
    merge(res_dir, final_plan_path)

    logger.info(f"Exploit search time: {exploit_searching_time:.2f}s")
    logger.info(f"Exploit analysis time: {exploit_analysis_time:.2f}s")

    # 6. *** NEW: Use LangChain 1.0 to analyze and summarize the results ***
    try:
        with open(final_plan_path) as f:
            exploit_data = json.load(f)
        
        if not exploit_data:
            logger.warning("Final plan file is empty. Cannot generate summary.")
            return
            
        logger.info("Generating AI-powered summary of the exploitation plan...")

        # Define the prompt and chain using LCEL
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior penetration testing expert. Your task is to analyze a JSON file containing potential exploits for a target application and provide a concise, actionable summary for a human operator. Respond ONLY with the requested JSON object."),
            ("human", "Analyze the following exploit data and provide your summary.\n\nJSON DATA:\n```json\n{exploit_json}\n```")
        ])
        
        # The chain pipes the prompt to the model, which is configured to output the structured Pydantic object
        summary_chain = summary_prompt | llm.with_structured_output(ExploitPlanSummary)
        
        # Invoke the chain with the data
        ai_summary = summary_chain.invoke({"exploit_json": json.dumps(exploit_data, indent=2)})

        # Print the structured output
        print("\n" + "="*30)
        print("🤖 AI-Generated Exploitation Plan")
        print("="*30)
        print(f"🔹 Summary: {ai_summary.summary}")
        print(f"🔹 Top CVE to Target: {ai_summary.top_cve}")
        print(f"🔹 Recommended Next Action: {ai_summary.recommended_action}")
        print("="*30 + "\n")

    except FileNotFoundError:
        logger.error(f"Final plan file not found at {final_plan_path}. Cannot generate summary.")
    except Exception as e:
        logger.error(f"An error occurred during AI summary generation: {e}")


if __name__ == "__main__":
    main()