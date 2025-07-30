# Remember to install langchain: pip install -qU "langchain[openai]"

import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Get the API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_LEY"] = getpass.getpass("Enter API key for OpenAU:")

# Simple Hello World Example
print ("Invoking the model with a simple text string")
model = init_chat_model("gpt-4o-mini", model_provider="openai")
response = model.invoke("Hello, world!")
print (response.content)

# Simple Messaages Example
print ("Invoking using System and Human Messages")
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]
model.invoke(messages)
for token in model.stream(messages):
    print(token.content, end="|")

# Simple Template Example
print ("\nInvoking with a Template Example")
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
prompt
prompt.to_messages()
response = model.invoke(prompt)
print(response.content)
