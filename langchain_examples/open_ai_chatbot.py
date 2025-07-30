import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

model = init_chat_model("gpt-4o-mini", model_provider="openai")

print ("Human Message Example")
response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
print (response.content)

print ("Demonstration of lack of state")
response = model.invoke([HumanMessage(content="What's my name?")])
print (response.content)
