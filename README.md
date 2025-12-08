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
(comming soon)

## Example usage
Currently the example scripts is in the [example](https://github.com/quanghona/agent_design_pattern/tree/master/example) folder.

The examples originally use LLMs hosted on local machine (ChatOllama for langchain, Ollama for llamaindex,...). To run example with cloud providers such as OpenAI, Anthropic, etc., you can setup the API key, initiate other appropriate LLM objects (e.g. ChatOpenAI) and pass to the Chain object and use pattern as normal.

## Disclaimer
Project is in initial development phase. All components are subject to change significantly.

## Plan:
- v0.1 (current): restructure project for generalization, modularization and integrate with LLM packages in the market
- v0.2: putting RAG techniques to PromptEnhancer class series in integration packages
- guardrails classes: PII, toxic, harmful, ...
- v0.3: apply HITL system wide. This approach requires interrupt mechanism and extensive support for comfirmation and intermidiate step and guidance from human, which will significantly change the component struture. Thus, this potentially requires a major restructure of the codebase again.

- v1.0: support multimodal agent
    + audio related agents: ASR TTS, STT,...
    + image related agents: OCR, captioning, image generation, etc
    + video related agents
    + other types of data: time-series, 3D, action as input data, etc
    + multimodal agent

Once the foundation layer becoming more stable, serious testings will be started soon to strengthen the low layers.
The doc for this project is also available soon with new versions release
