import os
import json
from utils.prompt import PentestAgentPrompt
from utils.dir_class import judge_class
from llama_index.core import (
    Settings,
    StorageContext,
    SimpleDirectoryReader,
    SummaryIndex,
    SimpleKeywordTableIndex,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import IndexNode
from llama_index.core.query_engine import RetrieverQueryEngine
import dotenv
from utils.vote import vote, get_final_res


class DocHandler:
    """
    该类负责处理数据目录的加载、索引和存储。
    """
    query_eng = None
    summary_dict = {}

    def __init__(self) -> None:
        pass

    def vul_analysis(self, cve:str, output_dir:str, vul_description:str):
        """
    对给定的 CVE 进行漏洞分析。
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
                for repo in subdirs:
                    repo_dir = os.path.join(code_dir, repo)
                    entries = [entry for entry in os.listdir(repo_dir) if not entry.startswith('.')]
                    if not os.path.exists(repo_dir) or not entries:
                        continue
                    
                    print(f"分析目录: {repo}")
                    code_reader = SimpleDirectoryReader(repo_dir, recursive=True, num_files_limit=10)
                    code_documents = code_reader.load_data()
                    repo_summary_index = SummaryIndex.from_documents(code_documents)
                    repo_query_engine = repo_summary_index.as_query_engine()

                    result["code"][code_source]["lang_class"][repo] = judge_class(repo_dir)
                    result["code"][code_source]["vul_type"][repo] = self.get_vul_category_from_code(cve, repo_query_engine, repo, output_dir)
                    
                    if repo == "Code_File":
                        result["code"][code_source]["exp_maturity"][repo] = "PoC"
                    else:
                        result["code"][code_source]["exp_maturity"][repo] = self.get_exp_maturity_analysis(cve, repo_query_engine, vul_description, repo, output_dir)
                    
                    result["code"][code_source]["isRemote"][repo] = self.get_isRemote_from_code(cve, repo_query_engine, repo, output_dir)
                    result["code"][code_source]["attack_complexity"][repo] = self.get_attack_complexity_from_code(cve, repo_query_engine, repo, output_dir)
        
        # 处理文档目录
        if os.path.exists(doc_dir) and os.listdir(doc_dir):
            doc_reader = SimpleDirectoryReader(doc_dir, recursive=True, num_files_limit=10)
            doc_documents = doc_reader.load_data()
            doc_summary_index = SummaryIndex.from_documents(doc_documents)
            doc_query_engine = doc_summary_index.as_query_engine()
            
            result["doc"]["vul_type"] = self.get_vul_category_from_doc(cve, doc_query_engine, output_dir)
            result["doc"]["isRemote"] = self.get_isRemote_from_doc(cve, doc_query_engine, output_dir)
            result["doc"]["attack_complexity"] = self.get_attack_complexity_from_doc(cve, doc_query_engine, output_dir)

        return result

    def get_vul_category_from_code(self, cve, query_engine, repo, output_dir):
        """ 从代码中提取漏洞类别 """
        categories = {
            "code_code_execution": PentestAgentPrompt.code_vul_code_execution_query,
            "code_privilege_escalation": PentestAgentPrompt.code_vul_privilege_escalation_query,
            "code_info_leak": PentestAgentPrompt.code_vul_information_leak_query,
            "code_bypass": PentestAgentPrompt.code_vul_bypass_query,
            "code_dos": PentestAgentPrompt.code_vul_denial_of_service_query,
        }
        features = {}
        try:
            for key, prompt in categories.items():
                features[key] = get_final_res(str(query_engine.query(prompt)))
        except Exception as e:
            print(f"查询漏洞类别时出错: {e}")
            for key in categories:
                features.setdefault(key, "None")
        
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

    def get_vul_category_from_doc(self, cve, query_engine, output_dir):
        """ 从文档中提取漏洞类别 """
        categories = {
            "doc_code_execution": PentestAgentPrompt.doc_vul_code_execution_query,
            "doc_privilege_escalation": PentestAgentPrompt.doc_vul_privilege_escalation_query,
            "doc_info_leak": PentestAgentPrompt.doc_vul_information_leak_query,
            "doc_bypass": PentestAgentPrompt.doc_vul_bypass_query,
            "doc_dos": PentestAgentPrompt.doc_vul_denial_of_service_query,
        }
        features = {}
        try:
            for key, prompt in categories.items():
                features[key] = get_final_res(str(query_engine.query(prompt)))
        except Exception as e:
            print(f"查询文档漏洞类别时出错: {e}")
            for key in categories:
                features.setdefault(key, "None")
        
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

    def get_exp_maturity_analysis(self, cve:str, query_engine, vul_impact:str, repo, output_dir):
        """ 分析漏洞利用成熟度 """
        features = {}
        try:
            features["code_description"] = str(query_engine.query(PentestAgentPrompt.code_description_query))
            features["code_poc"] = get_final_res(str(query_engine.query(PentestAgentPrompt.code_poc_query)))
            features["code_availability"] = str(query_engine.query(PentestAgentPrompt.code_availability_query))
            features["code_flexibility"] = get_final_res(str(query_engine.query(PentestAgentPrompt.code_flexibility_query)))
            features["code_functionality"] = get_final_res(str(query_engine.query(PentestAgentPrompt.code_functionality_query)))
        except Exception as e:
            print(f"分析漏洞利用成熟度时出错: {e}")

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

    def get_isRemote_from_code(self, cve:str, query_engine, repo, output_dir):
        """ 判断代码是否表明为远程利用 """
        is_remote = "None"
        try:
            is_remote = get_final_res(str(query_engine.query(PentestAgentPrompt.code_isRemote_query)))
        except Exception as e:
            print(f"判断是否远程利用时出错: {e}")
        
        path_segment = f"{cve}/" if "exploit" not in cve else ""
        filename = f"{output_dir}/{path_segment}isRemote_features_from_code_{repo}.json"
        with open(filename, 'w') as f:
            json.dump({"code_isRemote": is_remote}, f, indent=4)
        return is_remote
    
    def get_isRemote_from_doc(self, cve:str, query_engine, output_dir):
        """ 判断文档是否表明为远程利用 """
        is_remote = "None"
        try:
            is_remote = get_final_res(str(query_engine.query(PentestAgentPrompt.doc_isRemote_query)))
        except Exception as e:
            print(f"从文档判断是否远程利用时出错: {e}")
        
        path_segment = f"{cve}/" if "exploit" not in cve else ""
        filename = f"{output_dir}/{path_segment}isRemote_features_from_doc.json"
        with open(filename, 'w') as f:
            json.dump({"doc_isRemote": is_remote}, f, indent=4)
        return is_remote

    def get_attack_complexity_from_code(self, cve:str, query_engine, repo, output_dir):
        """ 从代码中分析攻击复杂度 """
        prompts = {
            "code_attack_evasion": (PentestAgentPrompt.code_attack_evasion_query, 4, True),
            "code_info_dependency": (PentestAgentPrompt.code_info_dependency_query, 2, False),
            "code_attack_condition": (PentestAgentPrompt.code_attack_condition_query, 2, False),
            "code_attack_probability": (PentestAgentPrompt.code_attack_probability_query, 4, True),
            "code_privilege_required": (PentestAgentPrompt.code_privilege_required_query, 2, False),
            "code_user_interaction": (PentestAgentPrompt.code_user_interaction_query, 2, False),
        }
        features, conf_scores = {}, {}
        try:
            for key, (prompt, votes, use_judge) in prompts.items():
                res, conf = vote(query_engine, prompt, no_vote=votes, use_judge=use_judge)
                features[key] = res
                conf_scores[key] = conf if conf else "3"
        except Exception as e:
            print(f"分析代码攻击复杂度时出错: {e}")
            for key in prompts:
                features.setdefault(key, "None")
                conf_scores.setdefault(key, "3")

        path_segment = f"{cve}/" if "exploit" not in cve else ""
        with open(f"{output_dir}/{path_segment}attack_complexity_features_from_code_{repo}.json", 'w') as f:
            json.dump(features, f, indent=4)
        with open(f"{output_dir}/{path_segment}attack_complexity_conf_score_from_code_{repo}.json", 'w') as f:
            json.dump(conf_scores, f, indent=4)
        return features
    
    def get_attack_complexity_from_doc(self, cve:str, query_engine, output_dir):
        """ 从文档中分析攻击复杂度 """
        prompts = {
            "doc_attack_evasion": PentestAgentPrompt.doc_attack_evasion_query,
            "doc_info_dependency": PentestAgentPrompt.doc_info_dependency_query,
            "doc_attack_condition": PentestAgentPrompt.doc_attack_condition_query,
            "doc_attack_probability": PentestAgentPrompt.doc_attack_probability_query,
            "doc_privilege_required": PentestAgentPrompt.doc_privilege_required_query,
            "doc_user_interaction": PentestAgentPrompt.doc_user_interaction_query,
        }
        features, conf_scores = {}, {}
        try:
            for key, prompt in prompts.items():
                res, conf = vote(query_engine, prompt)
                features[key] = res
                conf_scores[key] = conf if conf else "3"
        except Exception as e:
            print(f"分析文档攻击复杂度时出错: {e}")
            for key in prompts:
                features.setdefault(key, "None")
                conf_scores.setdefault(key, "3")

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
                    summary_index = SummaryIndex.from_documents(documents)
                    query_engine = summary_index.as_query_engine()
                    summary_txt = str(query_engine.query(summary_prompt))
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
    # 使用 Settings 全局配置 LLM 和嵌入模型
    Settings.llm = OpenAI(temperature=0, model="gpt-4o")
    Settings.embed_model = OpenAIEmbedding() # 明确指定嵌入模型

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