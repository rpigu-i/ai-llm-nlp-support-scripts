# use pip install -qU langchain[<your model>] to call the model

from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """Get the weather for a given city"""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# run the agent

agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
