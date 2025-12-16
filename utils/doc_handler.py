import os
import json
import logging
import time
import yaml
from utils.prompt import PentestAgentPrompt
from utils.dir_class import judge_class
# 移除 LlamaIndex 相关导入，只保留必要的
from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    load_index_from_storage,
)
from llama_index.core.schema import IndexNode
from llama_index.core.query_engine import RetrieverQueryEngine
import dotenv
from utils.vote import vote, get_final_res
# 使用 LangChain 1.0 的 ChatOpenAI，和 recon_agent_new.py 保持一致
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class DocHandler:
    """
    该类负责处理数据目录的加载、索引和存储。
    现在直接使用 LangChain 的 ChatOpenAI，不再使用 LlamaIndex 的查询引擎。
    """
    query_eng = None
    summary_dict = {}
    llm = None

    def __init__(self) -> None:
        # 初始化 LLM 配置，使用和 recon_agent_new.py 相同的方式
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM，使用 LangChain 的 ChatOpenAI，和 recon_agent_new.py 保持一致"""
        try:
            # 加载配置文件
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 获取模型配置，优先使用 planning 配置，否则使用 cve 配置
            if 'runtime' in config and 'planning' in config['runtime']:
                model_name = config['runtime']['planning'].get('model', 'gpt-4o-mini')
            elif 'cve' in config:
                model_name = config['cve'].get('model', 'gpt-4o-mini')
            else:
                model_name = 'gpt-4o-mini'
            
            # 获取超时配置
            timeout = config.get('models', {}).get('openai', {}).get('timeout', 180)
            
            # 使用 LangChain 的 ChatOpenAI，和 recon_agent_new.py 保持一致
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                max_retries=3,  # 设置最大重试次数，和 recon_agent_new.py 保持一致
                request_timeout=timeout  # 设置超时时间
            )
            logging.info(f"DocHandler LLM initialized successfully: {model_name} (timeout={timeout}s, max_retries=3)")
        except Exception as e:
            logging.error(f"Failed to initialize LLM in DocHandler: {e}")
            # 如果初始化失败，使用默认配置
            logging.warning("Falling back to default LLM configuration")
            try:
                self.llm = ChatOpenAI(
                    model='gpt-4o-mini',
                    temperature=0,
                    max_retries=3,
                    request_timeout=180
                )
            except Exception as fallback_error:
                logging.error(f"Fallback LLM initialization also failed: {fallback_error}")
                raise

    def _read_code_files(self, repo_dir: str) -> str:
        """读取代码目录中的所有文件内容，返回合并后的文本"""
        code_texts = []
        try:
            # 使用 SimpleDirectoryReader 读取文件（仅用于读取文件，不用于索引）
            reader = SimpleDirectoryReader(repo_dir, recursive=True, num_files_limit=10)
            documents = reader.load_data()
            for doc in documents:
                if doc.text:
                    code_texts.append(doc.text)
        except Exception as e:
            logging.warning(f"Failed to read code files from {repo_dir}: {e}")
        return "\n\n".join(code_texts)

    def _read_doc_files(self, doc_dir: str) -> str:
        """读取文档目录中的所有文件内容，返回合并后的文本"""
        doc_texts = []
        try:
            reader = SimpleDirectoryReader(doc_dir, recursive=True, num_files_limit=10)
            documents = reader.load_data()
            for doc in documents:
                if doc.text:
                    doc_texts.append(doc.text)
        except Exception as e:
            logging.warning(f"Failed to read doc files from {doc_dir}: {e}")
        return "\n\n".join(doc_texts)

    def _query_llm(self, prompt: str, context: str = "") -> str:
        """直接使用 LangChain ChatOpenAI 查询 LLM"""
        try:
            # 构建完整的 prompt，包含上下文
            if context:
                full_prompt = f"{context}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # 使用 LangChain 的 ChatOpenAI 直接调用
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            return response.content
        except Exception as e:
            logging.error(f"LLM query failed: {e}")
            raise

    def vul_analysis(self, cve:str, output_dir:str, vul_description:str):
        """
        对给定的 CVE 进行漏洞分析。
        现在直接使用 LangChain 的 ChatOpenAI，不再使用 LlamaIndex 的查询引擎。
        """
        if "exploit" not in cve:
            doc_dir = f"{output_dir}/{cve}/Google"
            code_dirs = {
                "ExploitDB": f"{output_dir}/{cve}/ExploitDB",
                "GitHub": f"{output_dir}/{cve}/GitHub"
            }
        else:
            doc_dir = f"{output_dir}/Google"
            code_dirs = {
                "ExploitDB": f"{output_dir}/ExploitDB",
                "GitHub": f"{output_dir}/GitHub"
            }
        
        result = {"code": {}, "doc": {}}
        
        # 处理代码目录 (ExploitDB 和 GitHub)
        for code_source, code_dir in code_dirs.items():
            if os.path.exists(code_dir) and os.listdir(code_dir):
                if code_source not in result["code"]:
                    result["code"][code_source] = {
                        "lang_class": {}, "vul_type": {}, "exp_maturity": {},
                        "isRemote": {}, "attack_complexity": {}
                    }
                
                subdirs = [f.name for f in os.scandir(code_dir) if f.is_dir()]
                # 添加次数限制：最多处理 50 个 repo，Code_File 只处理一次
                max_repos = 50
                code_file_processed = False
                processed_count = 0
                
                for repo in subdirs:
                    # 检查是否达到最大处理数量
                    if processed_count >= max_repos:
                        logging.info(f"达到最大处理数量限制 ({max_repos})，停止处理更多 repo")
                        break
                    
                    # 如果是 Code_File 且已经处理过，跳过
                    if repo == "Code_File" and code_file_processed:
                        logging.info(f"Code_File 已处理过，跳过重复处理")
                        continue
                    
                    repo_dir = os.path.join(code_dir, repo)
                    entries = [entry for entry in os.listdir(repo_dir) if not entry.startswith('.')]
                    if not os.path.exists(repo_dir) or not entries:
                        continue
                    
                    logging.info(f"分析目录: {repo} (CVE: {cve}, Source: {code_source})")
                    # 读取代码文件内容
                    code_context = self._read_code_files(repo_dir)

                    lang_class = judge_class(repo_dir)
                    result["code"][code_source]["lang_class"][repo] = lang_class
                    logging.info(f"  - 语言类型: {lang_class}")
                    
                    vul_type = self.get_vul_category_from_code(cve, code_context, repo, output_dir)
                    result["code"][code_source]["vul_type"][repo] = vul_type
                    logging.info(f"  - 漏洞类型: {vul_type}")
                    
                    if repo == "Code_File":
                        result["code"][code_source]["exp_maturity"][repo] = "PoC"
                        code_file_processed = True
                        logging.info(f"  - 利用成熟度: PoC (Code_File 默认值)")
                    else:
                        exp_maturity = self.get_exp_maturity_analysis(cve, code_context, vul_description, repo, output_dir)
                        result["code"][code_source]["exp_maturity"][repo] = exp_maturity
                        logging.info(f"  - 利用成熟度: {exp_maturity}")
                    
                    isRemote = self.get_isRemote_from_code(cve, code_context, repo, output_dir)
                    result["code"][code_source]["isRemote"][repo] = isRemote
                    logging.info(f"  - 是否远程利用: {isRemote}")
                    
                    attack_complexity = self.get_attack_complexity_from_code(cve, code_context, repo, output_dir)
                    result["code"][code_source]["attack_complexity"][repo] = attack_complexity
                    logging.info(f"  - 攻击复杂度: {attack_complexity}")
                    
                    logging.info(f"完成分析目录: {repo}")
                    processed_count += 1
        
        # 处理文档目录
        if os.path.exists(doc_dir) and os.listdir(doc_dir):
            # 读取文档文件内容
            doc_context = self._read_doc_files(doc_dir)
            
            result["doc"]["vul_type"] = self.get_vul_category_from_doc(cve, doc_context, output_dir)
            result["doc"]["isRemote"] = self.get_isRemote_from_doc(cve, doc_context, output_dir)
            result["doc"]["attack_complexity"] = self.get_attack_complexity_from_doc(cve, doc_context, output_dir)

        return result

    def get_vul_category_from_code(self, cve, code_context, repo, output_dir):
        """ 从代码中提取漏洞类别，直接使用 LangChain ChatOpenAI """
        categories = {
            "code_code_execution": PentestAgentPrompt.code_vul_code_execution_query,
            "code_privilege_escalation": PentestAgentPrompt.code_vul_privilege_escalation_query,
            "code_info_leak": PentestAgentPrompt.code_vul_information_leak_query,
            "code_bypass": PentestAgentPrompt.code_vul_bypass_query,
            "code_dos": PentestAgentPrompt.code_vul_denial_of_service_query,
        }
        features = {}
        total_categories = len(categories)
        current_index = 0
        # 为每个查询添加独立的错误处理，避免一个失败影响其他查询
        for key, prompt in categories.items():
            current_index += 1
            try:
                logging.info(f"    分析漏洞类别 ({current_index}/{total_categories}): {key}")
                # 直接使用 LangChain ChatOpenAI 查询
                response = self._query_llm(prompt, code_context)
                features[key] = get_final_res(response)
                logging.info(f"    结果: {key} = {features[key]}")
                # 添加延迟，避免请求频率过高导致限流
                time.sleep(1)  # 每次查询后等待1秒
            except Exception as e:
                logging.warning(f"查询漏洞类别 {key} 时出错 (CVE: {cve}, Repo: {repo}): {e}")
                features[key] = "None"  # 设置默认值，继续处理下一个
                # 即使失败也添加延迟，避免连续失败导致更严重的限流
                time.sleep(2)  # 失败后等待更长时间
        
        path_segment = f"{cve}/" if "exploit" not in cve else ""
        filename = f"{output_dir}/{path_segment}vul_impact_features_from_code_{repo}.json"
        with open(filename, 'w') as f:
            json.dump(features, f, indent=4)

        if features.get("code_code_execution") == "True": return "Code Execution"
        if features.get("code_privilege_escalation") == "True": return "Privilege Escalation"
        if features.get("code_info_leak") == "True": return "Information Leak"
        if features.get("code_bypass") == "True": return "Bypass"
        if features.get("code_dos") == "True": return "Denial of Service"
        return "Unknown"

    def get_vul_category_from_doc(self, cve, doc_context, output_dir):
        """ 从文档中提取漏洞类别，直接使用 LangChain ChatOpenAI """
        categories = {
            "doc_code_execution": PentestAgentPrompt.doc_vul_code_execution_query,
            "doc_privilege_escalation": PentestAgentPrompt.doc_vul_privilege_escalation_query,
            "doc_info_leak": PentestAgentPrompt.doc_vul_information_leak_query,
            "doc_bypass": PentestAgentPrompt.doc_vul_bypass_query,
            "doc_dos": PentestAgentPrompt.doc_vul_denial_of_service_query,
        }
        features = {}
        # 为每个查询添加独立的错误处理，避免一个失败影响其他查询
        for key, prompt in categories.items():
            try:
                # 直接使用 LangChain ChatOpenAI 查询
                response = self._query_llm(prompt, doc_context)
                features[key] = get_final_res(response)
                # 添加延迟，避免请求频率过高导致限流
                time.sleep(1)  # 每次查询后等待1秒
            except Exception as e:
                logging.warning(f"查询文档漏洞类别 {key} 时出错 (CVE: {cve}): {e}")
                features[key] = "None"  # 设置默认值，继续处理下一个
                # 即使失败也添加延迟，避免连续失败导致更严重的限流
                time.sleep(2)  # 失败后等待更长时间
        
        path_segment = f"{cve}/" if "exploit" not in cve else ""
        filename = f"{output_dir}/{path_segment}vul_impact_features_from_doc.json"
        with open(filename, 'w') as f:
            json.dump(features, f, indent=4)

        if features.get("doc_code_execution") == "True": return "Code Execution"
        if features.get("doc_privilege_escalation") == "True": return "Privilege Escalation"
        if features.get("doc_info_leak") == "True": return "Information Leak"
        if features.get("doc_bypass") == "True": return "Bypass"
        if features.get("doc_dos") == "True": return "Denial of Service"
        return "Unknown"

    def get_exp_maturity_analysis(self, cve:str, code_context:str, vul_impact:str, repo, output_dir):
        """ 分析漏洞利用成熟度，直接使用 LangChain ChatOpenAI """
        features = {}
        # 为每个查询添加独立的错误处理
        queries = {
            "code_description": (PentestAgentPrompt.code_description_query, str),
            "code_poc": (PentestAgentPrompt.code_poc_query, get_final_res),
            "code_availability": (PentestAgentPrompt.code_availability_query, str),
            "code_flexibility": (PentestAgentPrompt.code_flexibility_query, get_final_res),
            "code_functionality": (PentestAgentPrompt.code_functionality_query, get_final_res),
        }
        
        for key, (prompt, processor) in queries.items():
            try:
                # 直接使用 LangChain ChatOpenAI 查询
                response = self._query_llm(prompt, code_context)
                features[key] = processor(response)
                # 添加延迟，避免请求频率过高导致限流
                time.sleep(1)
            except Exception as e:
                logging.warning(f"分析漏洞利用成熟度 {key} 时出错 (CVE: {cve}, Repo: {repo}): {e}")
                # 设置合理的默认值
                if key == "code_poc":
                    features[key] = "False"
                elif key == "code_flexibility" or key == "code_functionality":
                    features[key] = "False"
                else:
                    features[key] = "None"
                # 失败后等待更长时间
                time.sleep(2)

        exp_maturity = "None"
        if features.get("code_poc") == "True":
            exp_maturity = "PoC"
            if (features.get("code_availability") == "True" or features.get("code_flexibility") == "True") and features.get("code_functionality") == "True":
                exp_maturity = "Exploit"
        
        features["exp_maturity"] = exp_maturity
        
        path_segment = f"{cve}/" if "exploit" not in cve else ""
        filename = f"{output_dir}/{path_segment}exp_maturity_features_{repo}.json"
        with open(filename, 'w') as f:
            json.dump(features, f, indent=4)

        return exp_maturity

    def get_isRemote_from_code(self, cve:str, code_context:str, repo, output_dir):
        """ 判断代码是否表明为远程利用，直接使用 LangChain ChatOpenAI """
        is_remote = "None"
        try:
            # 直接使用 LangChain ChatOpenAI 查询
            response = self._query_llm(PentestAgentPrompt.code_isRemote_query, code_context)
            is_remote = get_final_res(response)
            # 添加延迟，避免请求频率过高导致限流
            time.sleep(1)
        except Exception as e:
            logging.warning(f"判断是否远程利用时出错 (CVE: {cve}, Repo: {repo}): {e}")
            is_remote = "None"  # 确保返回默认值
            # 失败后等待更长时间
            time.sleep(2)
        
        path_segment = f"{cve}/" if "exploit" not in cve else ""
        filename = f"{output_dir}/{path_segment}isRemote_features_from_code_{repo}.json"
        try:
            with open(filename, 'w') as f:
                json.dump({"code_isRemote": is_remote}, f, indent=4)
        except Exception as e:
            logging.warning(f"保存 isRemote 特征文件失败 (CVE: {cve}, Repo: {repo}): {e}")
        return is_remote
    
    def get_isRemote_from_doc(self, cve:str, doc_context:str, output_dir):
        """ 判断文档是否表明为远程利用，直接使用 LangChain ChatOpenAI """
        is_remote = "None"
        try:
            # 直接使用 LangChain ChatOpenAI 查询
            response = self._query_llm(PentestAgentPrompt.doc_isRemote_query, doc_context)
            is_remote = get_final_res(response)
            # 添加延迟，避免请求频率过高导致限流
            time.sleep(1)
        except Exception as e:
            logging.warning(f"从文档判断是否远程利用时出错 (CVE: {cve}): {e}")
            is_remote = "None"  # 确保返回默认值
            # 失败后等待更长时间
            time.sleep(2)
        
        path_segment = f"{cve}/" if "exploit" not in cve else ""
        filename = f"{output_dir}/{path_segment}isRemote_features_from_doc.json"
        try:
            with open(filename, 'w') as f:
                json.dump({"doc_isRemote": is_remote}, f, indent=4)
        except Exception as e:
            logging.warning(f"保存 isRemote 文档特征文件失败 (CVE: {cve}): {e}")
        return is_remote

    def get_attack_complexity_from_code(self, cve:str, code_context:str, repo, output_dir):
        """ 从代码中分析攻击复杂度，直接使用 LangChain ChatOpenAI """
        prompts = {
            "code_attack_evasion": (PentestAgentPrompt.code_attack_evasion_query, 4, True),
            "code_info_dependency": (PentestAgentPrompt.code_info_dependency_query, 2, False),
            "code_attack_condition": (PentestAgentPrompt.code_attack_condition_query, 2, False),
            "code_attack_probability": (PentestAgentPrompt.code_attack_probability_query, 4, True),
            "code_privilege_required": (PentestAgentPrompt.code_privilege_required_query, 2, False),
            "code_user_interaction": (PentestAgentPrompt.code_user_interaction_query, 2, False),
        }
        features, conf_scores = {}, {}
        # 为每个查询添加独立的错误处理
        for key, (prompt, votes, use_judge) in prompts.items():
            try:
                # 使用修改后的 vote 函数，传入 LLM 而不是 query_engine
                res, conf = vote(self.llm, prompt, code_context, no_vote=votes, use_judge=use_judge)
                features[key] = res
                conf_scores[key] = conf if conf else "3"
                # 添加延迟，避免请求频率过高导致限流
                time.sleep(1)
            except Exception as e:
                logging.warning(f"分析代码攻击复杂度 {key} 时出错 (CVE: {cve}, Repo: {repo}): {e}")
                features[key] = "None"
                conf_scores[key] = "3"
                # 失败后等待更长时间
                time.sleep(2)

        path_segment = f"{cve}/" if "exploit" not in cve else ""
        with open(f"{output_dir}/{path_segment}attack_complexity_features_from_code_{repo}.json", 'w') as f:
            json.dump(features, f, indent=4)
        with open(f"{output_dir}/{path_segment}attack_complexity_conf_score_from_code_{repo}.json", 'w') as f:
            json.dump(conf_scores, f, indent=4)
        return features
    
    def get_attack_complexity_from_doc(self, cve:str, doc_context:str, output_dir):
        """ 从文档中分析攻击复杂度，直接使用 LangChain ChatOpenAI """
        prompts = {
            "doc_attack_evasion": PentestAgentPrompt.doc_attack_evasion_query,
            "doc_info_dependency": PentestAgentPrompt.doc_info_dependency_query,
            "doc_attack_condition": PentestAgentPrompt.doc_attack_condition_query,
            "doc_attack_probability": PentestAgentPrompt.doc_attack_probability_query,
            "doc_privilege_required": PentestAgentPrompt.doc_privilege_required_query,
            "doc_user_interaction": PentestAgentPrompt.doc_user_interaction_query,
        }
        features, conf_scores = {}, {}
        # 为每个查询添加独立的错误处理
        for key, prompt in prompts.items():
            try:
                # 使用修改后的 vote 函数，传入 LLM 而不是 query_engine
                res, conf = vote(self.llm, prompt, doc_context)
                features[key] = res
                conf_scores[key] = conf if conf else "3"
                # 添加延迟，避免请求频率过高导致限流
                time.sleep(1)
            except Exception as e:
                logging.warning(f"分析文档攻击复杂度 {key} 时出错 (CVE: {cve}): {e}")
                features[key] = "None"
                conf_scores[key] = "3"
                # 失败后等待更长时间
                time.sleep(2)

        path_segment = f"{cve}/" if "exploit" not in cve else ""
        with open(f"{output_dir}/{path_segment}attack_complexity_features_from_doc.json", 'w') as f:
            json.dump(features, f, indent=4)
        with open(f"{output_dir}/{path_segment}attack_complexity_conf_score_from_doc.json", 'w') as f:
            json.dump(conf_scores, f, indent=4)
        return features

    def create_index(self, topic_dir:str, summary_prompt:str, keyword:str):
        """ 创建或加载关键字索引 """
        keyword_index_dir = os.path.join(os.getenv("INDEX_STORAGE_DIR"), "keyword_repos", keyword)
        
        if not os.path.exists(keyword_index_dir):
            repo_index_nodes = []
            list_subfolders = [f.path for f in os.scandir(topic_dir) if f.is_dir()]
            
            for i, repo_dir in enumerate(list_subfolders):
                if any(not item.startswith('.') for item in os.listdir(repo_dir)):
                    print(f"创建索引: {repo_dir}")
                    reader = SimpleDirectoryReader(repo_dir, recursive=True)
                    documents = reader.load_data()
                    # 直接使用 LangChain ChatOpenAI 生成摘要
                    code_context = "\n\n".join([doc.text for doc in documents if doc.text])
                    summary_txt = self._query_llm(summary_prompt, code_context)
                    self.summary_dict[repo_dir] = summary_txt
                    
                    repo_index_node = IndexNode(
                        text=summary_txt,
                        metadata={"repo path": str(repo_dir)},
                        index_id=str(i)
                    )
                    repo_index_nodes.append(repo_index_node)
            
            # 使用 `nodes` 参数代替已弃用的 `objects`
            repos_keyword_index = SimpleKeywordTableIndex(nodes=repo_index_nodes)
            # 简化 StorageContext 的创建
            repos_keyword_index.storage_context.persist(persist_dir=keyword_index_dir)
        else:
            print(f"加载已有索引: {keyword_index_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=keyword_index_dir)
            repos_keyword_index = load_index_from_storage(storage_context)
        
        # 移除已弃用的 `verbose` 参数
        keyword_retriever = repos_keyword_index.as_retriever()
        
        # 更新 RetrieverQueryEngine 的实例化方式
        self.query_eng = RetrieverQueryEngine(retriever=keyword_retriever)

    def query(self, query_content):
        """ 执行查询 """
        if self.query_eng:
            return self.query_eng.query(query_content)
        return "查询引擎未初始化。"

def main():
    dotenv.load_dotenv()
    # 使用 LangChain ChatOpenAI
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    doc_dir = 'resources/CVE-2019-1609/GitHub'
    summary_prompt = PentestAgentPrompt.repo_summary
    
    doc_handler = DocHandler()
    doc_handler.create_index(doc_dir, summary_prompt, "CVE-2019-1609_github")
    
    service = "Cisco"
    version = ""
    query_content = (
        f"清晰列出所有包含适用于 {service} 版本 {version} 的漏洞利用 PoC 的相关代码库路径，"
        f"并为每个选择提供支持理由。要比较版本，可以按从左到右的顺序比较版本号。"
        f"例如，7.4.0 是比 8.2.3 更早的版本。通过检查 {version} 是否在漏洞利用的适用版本范围内来进行选择。"
        "仅考虑上下文中提到的路径。"
    )
    res = doc_handler.query(query_content)
    print("查询结果:")
    print(res)

if __name__ == "__main__":
    main()
