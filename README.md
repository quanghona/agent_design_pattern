# AI Agent pattern (AAP)
AI Agent Pattern aim to provide orchestration logic for agents. This project aim to target orchestration between agents. The LLM manipulation is delegated to other libraries such as langchain, llamaindex, transformers, etc. Thus, it will have integration with those frameworks beside the core logic.

## Getting started

### Langchain
```bash
pip install aap_langchain
```

### LlamaIndex
```bash
pip install aap_llamaindex
```

### Transformers
```bash
pip install aap_transformers
```

### DsPy
```bash
pip install aap_dspy
```

For the The DSPy implementation, the adapter is a little bit different from others. The library use [Signature](https://dspy.ai/api/signatures/Signature/) to deal with the fields. To be able to use with the AgentMessage in our library, we need to have a another adapter class stay in between the message and the signature. In the input of the flow, the adapter converts the message to the signature before passing to the DSPy's predictor. In the output of the flow, the adapter converts the signature to the message before returning control to the flow.

From another perspective, we can call the LM directly as stated in the [dspy doc](https://dspy.ai/learn/programming/language_models/#calling-the-lm-directly). In fact the logic will become similar to other integration packages when implementing the flow. But it will be hard to make use of the pattern and follow the principle behind the DSPy framework - signature and prompt programming, so it is considered anti-pattern and not recommended.

For a simple example, we implement with following steps:
1. Define the signature: either by using string or subclass of dspy.Signature
2. Define the adapter: by subclassing the BaseSignatureAdapter. The adapter should implement the `msg2sig` and `sig2msg` methods
3. Create chain object: we will pass in the signature type and the adapter object to the chain constructor. There are 2 cases:
    - We can use dspy.configure to setup the model and use the default context
    - The chain have the with_lm method to setup for specific chain. This option enable use to use multiple LMs in one flow.
4. The remaining logic is similar to other integration packages: define agents, flows, etc

## Example usage
Currently the example scripts is in the [example](https://github.com/quanghona/agent_design_pattern/tree/master/example) folder.

The examples originally use LLMs hosted on local machine (ChatOllama for langchain, Ollama for llamaindex,...). To run example with cloud providers such as OpenAI, Anthropic, etc., you can setup the API key, initiate other appropriate LLM objects (e.g. ChatOpenAI) and pass to the Chain object and use pattern as normal.

## Disclaimer
Project is in initial development phase. All components are subject to change significantly.

## Plan:
- v0.1: restructure project for generalization, modularization and integrate with LLM packages in the market
- v.0.1.2 (current): start adding dspy to the support list
- v0.2: putting RAG techniques to PromptEnhancer class series in integration packages
- v0.2.1: Turn on support for [toon format](https://github.com/toon-format/toon-python) of the prompt enhancers when the library version bump to v1.0.0
- v0.3: guardrails classes: PII, toxic, harmful, ...
- v0.4: apply HITL system wide. This approach requires interrupt mechanism and extensive support for comfirmation and intermidiate step and guidance from human, which will significantly change the component struture. Thus, this potentially requires a major restructure of the codebase again.

- Unified token usage data in the output

- The inference for the diffusion models (i.e. dLLM) will also be planned in the v0.x alongside with current auto-regressive models.

- v1.0: support multimodal agent
    + audio related agents: ASR TTS, STT,...
    + image related agents: OCR, captioning, image generation, etc
    + video related agents
    + other types of data: time-series, 3D, action as input data, etc
    + multimodal agent

Once the foundation layer becoming more stable, serious testings will be started soon to strengthen the low layers.
The doc for this project is also available soon with new versions release
