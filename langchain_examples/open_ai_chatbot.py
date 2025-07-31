# remember to install langgraph in addition to langchain e.g. pip install -U langgraph

import getpass
import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Add state class
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Define the function that leverage templates for model calls
def call_template_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response} 

# Define the function that leverage templates with State class
def call_template_state_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

# Output a query result
def output_query(query, config):
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

print ("Simple Open AI Chat example with LangGraph")
llm = ChatOpenAI()
llm.invoke("Hello, world!")

print ("Human Message Example")
response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
print (response.content)

print ("Demonstration of lack of state")
response = model.invoke([HumanMessage(content="What's my name?")])
print (response.content)

print ("Example passing conversation history")
response = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you todat?"),
        HumanMessage(content="What's my name?"),
    ]
)
print (response.content)

print ("Example using LangGraph")

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Include a thread to denote a unqiue conversation 
config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."
output_query(query, config)

query = "What's my name?"
output_query(query, config)

print ("Demonstration of switching threads")

# New thread ID
config = {"configurable": {"thread_id": "abc234"}}
output_query(query, config)

print ("Demonstration of swithcing back to original thread")

# Switch back to our original thread ID
config = {"configurable": {"thread_id": "abc123"}}
output_query(query, config)

print ("Prompt templates")
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_template_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc345"}}
query = "Hi! I'm Jim."
output_query(query, config)

query = "What is my name?"
output_query(query, config)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assisant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_template_state_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

print ("Example in Spanish")
config = {"configurable": {"thread_id": "abc456"}}
query = "Hi! I'm Bob."
language = "Spanish"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

print ("Example omitting params to demonstrate persistent state")
query = "What is my name?"
output_query(query, config)

