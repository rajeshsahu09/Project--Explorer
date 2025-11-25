from llm_integration.llm_client import OpenLLM

llm = OpenLLM()
response = llm.invoke([{"role": "user", "content": "Do you have my file system access?"}])
print(response)