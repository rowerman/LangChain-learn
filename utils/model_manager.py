'''import os
import logging
import yaml
from typing import Dict, Any, Optional
from utils.llm_factory import llm_manager, create_llm_from_config

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
        self.config_path = config_path
        self.config = self._load_config()
        self.initialized_models = {}

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _process_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_val = os.getenv(env_var)
                if env_val is None:
                    logger.warning(f"Env var {env_var} not found")
                    env_val = ""
                processed[key] = env_val
            else:
                processed[key] = value
        return processed

    def get_model(self, model_name: str = "openai"):
        if model_name in self.initialized_models:
            return self.initialized_models[model_name]

        model_cfgs = self.config.get("models", {})
        if model_name not in model_cfgs:
            raise ValueError(f"Model '{model_name}' not found in config")

        cfg = self._process_environment_variables(model_cfgs[model_name])
        cfg['name'] = model_name
        llm = create_llm_from_config(cfg)
        self.initialized_models[model_name] = llm
        return llm

    def get_runtime_model(self, agent_type: str) -> str:
        runtime = self.config.get("runtime", {})
        return runtime.get(agent_type, {}).get("model", "default")

model_manager = ModelManager()

def get_model(model_name: str = "default"):
    try:
        return model_manager.get_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        return None

def get_agent_model(agent_type: str):
    return model_manager.get_model(model_manager.get_runtime_model(agent_type))
    '''

    # +++ 请用这个完整的类替换掉 model_manager.py 中的原有内容 +++

import os
import logging
import yaml
from typing import Dict, Any, Optional
from utils.llm_factory import llm_manager, create_llm_from_config

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
        self.config_path = config_path
        self.config = self._load_config()
        self.initialized_models = {}

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _process_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_val = os.getenv(env_var)
                if env_val is None:
                    logger.warning(f"Env var {env_var} not found")
                    env_val = ""
                processed[key] = env_val
            else:
                processed[key] = value
        return processed

    def get_model(self, model_name: str = "openai"):
        """
        获取一个模型实例。
        此版本经过修改，可以查找 model 字段值与 model_name 匹配的配置，
        而不仅仅是匹配配置的主键名。
        """
        if model_name in self.initialized_models:
            return self.initialized_models[model_name]

        model_cfgs = self.config.get("models", {})
        
        # --- 这是核心修改部分 ---
        found_config = None
        # 首先尝试直接用 model_name 作为主键查找，保持向后兼容
        if model_name in model_cfgs:
            found_config = model_cfgs[model_name]
        else:
            # 如果没找到，则遍历所有配置，查找内部的 'model' 字段
            for config_key, config_details in model_cfgs.items():
                if isinstance(config_details, dict) and config_details.get("model") == model_name:
                    found_config = config_details
                    logger.info(f"Found model '{model_name}' under configuration key '{config_key}'.")
                    break
        
        if found_config is None:
            # 如果两种方式都找不到，则报错
            raise ValueError(f"Model '{model_name}' not found in config, either as a key or as a 'model' value.")
        # --- 修改结束 ---

        cfg = self._process_environment_variables(found_config)
        # 确保传递给工厂函数的配置中包含正确的模型名称
        if 'model' not in cfg:
            cfg['model'] = model_name
        
        llm = create_llm_from_config(cfg)
        self.initialized_models[model_name] = llm
        return llm

    def get_runtime_model(self, agent_type: str) -> str:
        runtime = self.config.get("runtime", {})
        return runtime.get(agent_type, {}).get("model", "default")

# --- 下面的部分保持不变 ---
model_manager = ModelManager()

def get_model(model_name: str = "default"):
    try:
        return model_manager.get_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        return None

def get_agent_model(agent_type: str):
    return model_manager.get_model(model_manager.get_runtime_model(agent_type))