from langchain_community.tools import ShellTool

shell_tool = ShellTool()
result = shell_tool.invoke("Who am i")
print(result)