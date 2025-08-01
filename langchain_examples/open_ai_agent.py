# In order to work with tavily install it via pip pip install -qU langchain-tavily
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Grab API Keys
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

def agent_streamer(input_message, config):
    for step in agent_executor.stream(
        {"messages": [input_message]}, config, stream_mode="values"
):
        step["messages"][-1].pretty_print() 

# Create a basic agent
memory = MemorySaver()
model = init_chat_model("openai:gpt-4o-mini")
search = TavilySearch(max_result=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Start using the model
print ("Start using the model")
query = "Good morning"
response = model.invoke([{"role": "user", "content": query}])
output = response.text()
print (output)

# Enable the model for tool calling
print ("Example with binding tools")
model_with_tools = model.bind_tools(tools)

query = "Good morning, again!"
response = model_with_tools.invoke([{"role": "user", "content": query}])
print (f"Message content: {response.text()}\n")
print (f"Tool calls: {response.tool_calls}")

# Tool invocation
print ("Example of how a tool can be invoked")
query = "Search for the weather in SF"
response = model_with_tools.invoke([{"role": "user", "content": query}])
print (f"Message content: {response.text()}\n")
print (f"Tool calls: {response.tool_calls}")


# Use the agent
print ("Basic example using the agent, bind_tools is handled under the hood")
config = {"configurable": {"thread_id": "abc123"}}
input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in SF.",
}
agent_streamer(input_message, config)

print ("Example asking the weather. This will run a search")
input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}
agent_streamer(input_message, config)


# Running the react agent
input_message = {"role": "user", "content": "Hi!"}
response = agent_executor.invoke({"messages": [input_message]}, config, stream_mode="values")

for message in response["messages"]:
    message.pretty_print()


# Stream tokens
print ("Streaming tokens from the search")
for step, metadata in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="messages"
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")
