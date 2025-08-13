from strands import Agent

# Agent will watch ./tools/ director for changes

agent = Agent(load_tools_from_directory=True)
response = agent("Use any tools you find in the tools directory")


