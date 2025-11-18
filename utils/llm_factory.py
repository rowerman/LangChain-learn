# +++ 文件: utils/llm_factory.py (最终稳定版) +++

import logging
from typing import Dict, Any
from abc import ABC, abstractmethod

# --- 核心改动：同时导入 LangChain 和 LlamaIndex 的组件 ---
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI  # 直接使用 LangChain 的 OpenAI 集成

# LlamaIndex 的组件仅用于其他不需要 invoke() 的地方
from llama_index.core.llms.llm import LLM as LlamaIndexLLM
from llama_index.llms.deepseek import DeepSeek

logger = logging.getLogger(__name__)

# --- 工厂逻辑重构 ---

def create_llm_from_config(config: Dict[str, Any]) -> BaseLanguageModel:
    """
    根据配置直接创建并返回一个 LangChain 兼容的 LLM 实例。
    这个版本会根据 provider 直接创建对应的 LangChain 兼容对象。
    """
    provider = config.get("provider", "openai").lower()
    
    if provider == "openai":
        try:
            # 直接创建 LangChain 的 ChatOpenAI 实例
            # 它天生就拥有 .invoke() 方法
            llm = ChatOpenAI(
                model=config.get("model"),
                temperature=config.get("temperature", 0.0),
                api_key=config.get("api_key"), # 会自动从环境变量 OPENAI_API_KEY 读取
                base_url=config.get("base_url"),
                request_timeout=config.get("timeout", 180)
            )
            logger.info(f"Successfully created LangChain ChatOpenAI instance for model: {config.get('model')}")
            return llm
        except Exception as e:
            logger.error(f"Failed to create ChatOpenAI instance: {e}")
            raise

    # 如果您需要支持 DeepSeek 且让它也兼容 LangChain，需要 LlamaIndex 转换
    # 注意：这要求 LlamaIndex 必须是新版本
    elif provider == "deepseek":
        try:
            # 对于 DeepSeek，我们仍然尝试 LlamaIndex -> LangChain 的转换
            deepseek_llm = DeepSeek(
                api_key=config['api_key'],
                model=config['model'],
                temperature=config.get('temperature', 0),
                timeout=config.get('timeout', 30),
            )
            # 这里的 .to_langchain() 仍然是潜在的失败点，如果 LlamaIndex 版本过低
            return deepseek_llm.to_langchain()
        except Exception as e:
            logger.error(f"Failed to create and convert DeepSeek LLM: {e}")
            raise
    
    else:
        raise ValueError(f"Unsupported LLM provider for LangChain compatibility: {provider}")

# --- 原有的 Manager 和 get_model/get_agent_model 逻辑保持不变 ---
# model_manager.py 会调用上面的 create_llm_from_config 函数，
# 所以我们不再需要复杂的工厂类和 Provider 类。

# 为了让 model_manager.py 能正常工作，我们保留一个 llm_manager 占位符
# 和一个空的 create_llm 函数（虽然它不会被 model_manager 调用）
llm_manager = None