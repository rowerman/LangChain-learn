import sys
import langchain

# 打印正在使用的 Python 解释器路径
print(f"Python Executable: {sys.executable}")

# 打印 langchain 库的安装路径和版本号
print(f"LangChain Path: {langchain.__file__}")
print(f"LangChain Version: {langchain.__version__}")

try:
    from langchain.agents import AgentExecutor
    print("\nSuccessfully imported AgentExecutor from langchain.agents")
except ImportError as e:
    print(f"\nFailed to import AgentExecutor. Error: {e}")
