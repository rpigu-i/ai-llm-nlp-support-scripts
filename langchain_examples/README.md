# LangChain Examples

LangChain examples are broken out into two groups:

1. AWS 
2. OpenAI

Depending on which examples you are working with, you will need the coresponding backend 
platforms configured. 


## OpenAI

In order to execute the OpenAI examples, configure:

1. A project e.g. "LangChainExamples" under the projects section of your OpenAI account at: https://platform.openai.com/settings/organization/projects

2. An API Key associated with the project. This can be created at: https://platform.openai.com/settings/organization/api-keys

Remember to then set the API Key in your environment, for example on the Mac:

`export OPENAI_API_KEY="your_openai_api_key_here"`

If you plan to trace the results to LangGraph, also remember to set the tracing environemnt variable to true, and include your API key:

```
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

## LangGraph Examples

In order to leverage the tracing feature, you will need to setup a LangGraph account.

Once complete, navigate to the Tracing project in the leftb hand menu and start a new one, then:

1. Generate a new API Key using the button on the Set up observability window

2. Set the Project Name e.g. `LangGraphExample`

3. Copy the environment variables, and `export` them. Be careful not to blow away your existing `OPENAI_API_KEY` you set already. Note, this should also include setting `LANGSMITH_TRACING` to true, and including your `LANGSMITH_API_KEY`. 

## Agent Example

The agent example uses Tavily to conduct searches on the Internet and process the data. In order to use the package, install it via pip

```
pip3 install -qU langchain-tavily
```

If you don't have a Tavily API key, you can create one here: https://auth.tavily.com/

Once you have the key, remember to `export` it into your environment.









