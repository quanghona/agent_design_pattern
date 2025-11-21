# Agent design pattern
Agent design pattern aim to provide orchestration logic for agents. This project aim to target orchestration between agents. The LLM manipulation is delegated to other libraries such as langchain, llamaindex, transformers, etc. Thus, it will have integration with those frameworks beside the core logic.


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
