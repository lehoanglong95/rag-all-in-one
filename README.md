# RAGtoolkit 

<div align="center">
  <p>
    <a href="https://www.linkedin.com/in/long-le-713b41111/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a> |
    <a href="https://github.com/lehoanglong95" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" /></a>
  </p>
</div>

## Introduction

RAGtoolkit is a guide to building Retrieval-Augmented Generation (RAG) applications. It offers a collection of tools, libraries, and frameworks for RAG systems, with explanations of key components and recommendations for effective implementation.

### RAG Architecture Diagram

![RAG Architecture](RAG%20Diagram.png)

## RAG Components ✅

| Component | Description |
|-----------|-------------|
| [📄 Document Ingestor](#document-ingestor-) | Tools for ingesting and processing raw documents. Document loaders, parsers, and preprocessing tools |
| [🤖 Agent Framework](#agent-framework) | End-to-end frameworks for building RAG applications. Unified solutions for RAG implementation |
| [📀 Database](#vector-database) | Databases optimized for storing and searching vector embeddings. Vector storage, similarity search, and indexing |
| [💻 LLM](#llm) | Large Language Models for generating responses. LLM providers and frameworks |
| [📝 Embedding](#embedding) | Models and services for creating text embeddings. Embedding models and APIs |
| [🖥️ LLM Observability](#llm-observability) | Tools for monitoring and analyzing LLM performance. Logging, tracing, and analytics |
| [📕 Prompt Techniques](#prompt-techniques) | Methods for effective prompt engineering. Prompt templates and frameworks |
| [🤔 Evaluation](#evaluation) | Tools for assessing RAG system performance. Metrics and evaluation frameworks |

## Document Ingestor ✅

Tools and libraries for ingesting various document formats, extracting text, and preparing data for further processing.

| Library | Description | Link |
|---------|-------------|------|
| Unstructured | Library for pre-processing and extracting content from raw documents | [GitHub](https://github.com/Unstructured-IO/unstructured) |
| Docling | Document processing tool that parses diverse formats with advanced PDF understanding and AI integrations | [GitHub](https://github.com/docling-project/docling) |
| PyMuPDF | A Python binding for MuPDF, offering fast PDF processing capabilities | [GitHub](https://github.com/pymupdf/PyMuPDF) |
| PyPDF | Library for reading and manipulating PDF files | [GitHub](https://github.com/py-pdf/pypdf) |
| PyPDF | Library for reading and manipulating PDF files | [GitHub](https://github.com/py-pdf/pypdf) |
| LangChain Document Loaders | Comprehensive set of document loaders for various file types | [GitHub](https://github.com/langchain-ai/langchain) |
| MegaParse | Versatile parser for text, PDFs, PowerPoint, and Word documents with lossless information extraction | [GitHub](https://github.com/QuivrHQ/MegaParse) |
| Adobe PDF Extract | A service provided by Adobe for extracting content from PDF documents | [Link](https://developer.adobe.com/document-services/docs/overview/legacy-documentation/pdf-extract-api/quickstarts/python/) |
| Azure AI Document Intelligence | A service provided by Azure for extracting content including text, tables, images from PDF documents | [Link](https://developer.adobe.com/document-services/docs/overview/legacy-documentation/pdf-extract-api/quickstarts/python/) |

## Agent Framework

End-to-end frameworks that provide integrated solutions for building RAG applications.

| Library | Description | Link |
|-----------|-------------|------|
| LangChain | Framework for building applications with LLMs and integrating with various data sources | [GitHub](https://github.com/langchain-ai/langchain) |
| LlamaIndex | Data framework for building RAG systems with structured data | [GitHub](https://github.com/jerryjliu/llama_index) |
| Haystack | End-to-end framework for building NLP pipelines | [GitHub](https://github.com/deepset-ai/haystack) |
| Pydantic AI | Agent Framework / shim to use Pydantic with LLMs | [GitHub](https://github.com/pydantic/pydantic-ai) |
| SmolAgents | A barebones library for agents | [GitHub](https://github.com/huggingface/smolagents) |
| txtai | Open-source embeddings database for semantic search and LLM workflows | [GitHub](https://github.com/neuml/txtai) |
| OpenAI Agent | A lightweight, powerful framework for multi-agent workflows | [GitHub](https://github.com/openai/openai-agents-python) |

## Vector Database 

Databases optimized for storing and efficiently searching vector embeddings/text documents.

| Database | Description | Link |
|----------|-------------|------|
| Pinecone | Managed vector database for semantic search | [Website](https://www.pinecone.io/) |
| Weaviate | Open-source vector search engine | [GitHub](https://github.com/weaviate/weaviate) |
| Milvus | Open-source vector database | [GitHub](https://github.com/milvus-io/milvus) |
| Qdrant | Vector similarity search engine | [GitHub](https://github.com/qdrant/qdrant) |
| Chroma | Open-source embedding database designed for RAG applications | [GitHub](https://github.com/chroma-core/chroma) |
| FAISS | Efficient similarity search library from Facebook AI Research | [GitHub](https://github.com/facebookresearch/faiss) |
| MongoDB | General-purpose document database | [Website](https://www.mongodb.com/) |
| Elasticsearch | Search and analytics engine that can store documents | [Website](https://www.elastic.co/) |
| LanceDB | Relational database with JSON support | [Website](https://lancedb.com/) |

## LLM

Large Language Models and platforms for generating responses based on retrieved context.

| LLM | Description | Link |
|-----|-------------|------|
| OpenAI API | Access to GPT models through API | [Website](https://platform.openai.com/) |
| Claude | Anthropic's Claude series of LLMs | [Website](https://www.anthropic.com/claude) |
| Hugging Face LLM Models| Platform for open-source NLP models | [Hugging Face](https://huggingface.co/collections/open-llm-leaderboard/open-llm-leaderboard-best-models-652d6c7965a4619fb5c27a03) |
| LLaMA | Meta's open-source large language model | [GitHub](https://github.com/facebookresearch/llama) |
| Mistral | Open-source and commercial models | [Website](https://mistral.ai/) |
| Cohere | API access to generative and embedding models | [Website](https://cohere.com/) |
| DeepSeek | Advanced large language models for various applications | [Website](https://www.deepseek.com/) |
| Qwen | Alibaba Cloud's large language model accessible via API | [Website](https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api) |
| Ollama | Run open-source LLMs locally | [GitHub](https://github.com/ollama/ollama) |

## Embedding

Models and services for creating vector representations of text.

| Embedding Solution | Description | Link |
|-------------------|-------------|------|
| OpenAI Embeddings | API for text-embedding-ada-002 and newer models | [Documentation](https://platform.openai.com/docs/guides/embeddings) |
| Sentence Transformers | Python framework for state-of-the-art sentence embeddings | [GitHub](https://github.com/UKPLab/sentence-transformers) |
| Cohere Embed | Specialized embedding models API | [Website](https://cohere.com/embed) |
| Hugging Face Embeddings | Various embedding models | [Hugging Face](https://huggingface.co/models?pipeline_tag=feature-extraction) |
| E5 Embeddings | Microsoft's text embeddings | [Hugging Face](https://huggingface.co/intfloat/e5-large-v2) |
| BGE Embeddings | BAAI general embeddings | [Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5) |

## LLM Observability

Tools for monitoring, analyzing, and improving LLM applications.


| Library | Description | Link | 🌟                                                                       |
|------|-------------|------|---------------------------------------------------------------------------|
| Langfuse | Open source LLM engineering platform | [GitHub](https://github.com/langfuse/langfuse) | ![GitHub stars](https://img.shields.io/github/stars/langfuse/langfuse) |
| Helicone | Open source LLM observability platform. One line of code to monitor, evaluate, and experiment | [GitHub](https://github.com/helicone/helicone) | ![GitHub stars](https://img.shields.io/github/stars/helicone/helicone) |
| Opik/Comet | Debug, evaluate, and monitor LLM applications with tracing, evaluations, and dashboards | [GitHub](https://github.com/comet-ml/opik) | ![GitHub stars](https://img.shields.io/github/stars/comet-ml/opik) |
| Phoenix/Arize | Open-source observability for LLM applications | [GitHub](https://github.com/Arize-ai/phoenix) | ![GitHub stars](https://img.shields.io/github/stars/Arize-ai/phoenix) |
| Lunary | The production toolkit for LLMs. Observability, prompt management and evaluations. | [GitHub](https://github.com/lunary-ai/lunary) | ![GitHub stars](https://img.shields.io/github/stars/lunary-ai/lunary) |
| Openlit | Open source platform for AI Engineering: OpenTelemetry-native LLM Observability, GPU Monitoring, Guardrails, Evaluations, Prompt Management, Vault, Playground | [GitHub](https://github.com/openlit/openlit) | ![GitHub stars](https://img.shields.io/github/stars/openlit/openlit) |
| Langtrace | OpenTelemetry-based observability tool for LLM applications with real-time tracing and metrics | [GitHub](https://github.com/Scale3-Labs/langtrace) | ![GitHub stars](https://img.shields.io/github/stars/Scale3-Labs/langtrace) |

## Prompt Techniques

Methods and frameworks for effective prompt engineering in RAG systems.

### Open Source Prompt Engineering Tools

| Library | Description | Link | 🌟 |
|------|-------------|------|-------|
| Prompt Engineering Guide | Comprehensive guide to prompt engineering | [GitHub](https://github.com/dair-ai/Prompt-Engineering-Guide) | ![GitHub stars](https://img.shields.io/github/stars/dair-ai/Prompt-Engineering-Guide) |
| DSPy | Framework for programming language models instead of prompting | [GitHub](https://github.com/stanfordnlp/dspy) | ![GitHub stars](https://img.shields.io/github/stars/stanfordnlp/dspy) |
| Guidance | Language for controlling LLMs | [GitHub](https://github.com/guidance-ai/guidance) | ![GitHub stars](https://img.shields.io/github/stars/guidance-ai/guidance) |
| LLMLingua | Prompt compression library for faster LLM inference | [GitHub](https://github.com/microsoft/LLMLingua) | ![GitHub stars](https://img.shields.io/github/stars/microsoft/LLMLingua) |
| Promptify | NLP task prompt generator for GPT, PaLM and other models | [GitHub](https://github.com/promptslab/Promptify) | ![GitHub stars](https://img.shields.io/github/stars/promptslab/Promptify) |
| PromptSource | Toolkit for creating and sharing natural language prompts | [GitHub](https://github.com/bigscience-workshop/promptsource) | ![GitHub stars](https://img.shields.io/github/stars/bigscience-workshop/promptsource) |
| Promptimizer | Library for optimizing prompts | [GitHub](https://github.com/hinthornw/promptimizer) | ![GitHub stars](https://img.shields.io/github/stars/hinthornw/promptimizer) |
| Selective Context | Context compression tool for doubling LLM content processing | [GitHub](https://github.com/liyucheng09/Selective_Context) | ![GitHub stars](https://img.shields.io/github/stars/liyucheng09/Selective_Context) |
| betterprompt | Testing suite for LLM prompts before production | [GitHub](https://github.com/stjordanis/betterprompt) | ![GitHub stars](https://img.shields.io/github/stars/stjordanis/betterprompt) |

### Documentation & Services

| Resource | Description | Link |
|----------|-------------|------|
| OpenAI Prompt Engineering | Official guide to prompt engineering from OpenAI | [Link](https://platform.openai.com/docs/guides/prompt-engineering) |
| LangChain Prompts | Templates and composition tools for prompts | [Link](https://python.langchain.com/docs/how_to/) |
| PromptPerfect | Tool for optimizing prompts | [Link](https://promptperfect.jina.ai/) |

## Evaluation

Tools and frameworks for assessing and improving RAG system performance.

| Tool | Description | Link | 🌟 |
|------|-------------|------|-------|
| FastChat | Open platform for training, serving, and evaluating LLM-based chatbots | [Github](https://github.com/lm-sys/fastchat) | ![GitHub stars](https://img.shields.io/github/stars/lm-sys/fastchat) |
| OpenAI Evals | Framework for evaluating LLMs and LLM systems | [GitHub](https://github.com/openai/evals) | ![GitHub stars](https://img.shields.io/github/stars/openai/evals) |
| RAGAS | Ultimate toolkit for evaluating and optimizing RAG systems | [GitHub](https://github.com/explodinggradients/ragas) | ![GitHub stars](https://img.shields.io/github/stars/explodinggradients/ragas) |
| Promptfoo | Open-source tool for testing and evaluating prompts | [GitHub](https://github.com/promptfoo/promptfoo) | ![GitHub stars](https://img.shields.io/github/stars/promptfoo/promptfoo) |
| DeepEval | Comprehensive evaluation library for LLM applications | [GitHub](https://github.com/confident-ai/deepeval) | ![GitHub stars](https://img.shields.io/github/stars/confident-ai/deepeval) |
| Giskard | Open-source evaluation and testing for ML & LLM systems | [Github](https://github.com/giskard-ai/giskard) | ![GitHub stars](https://img.shields.io/github/stars/giskard-ai/giskard) |
| PromptBench | Unified evaluation framework for large language models | [Github](https://github.com/microsoft/promptbench) | ![GitHub stars](https://img.shields.io/github/stars/microsoft/promptbench) |
| TruLens | Evaluation and tracking for LLM experiments with RAG-specific metrics | [GitHub](https://github.com/truera/trulens) | ![GitHub stars](https://img.shields.io/github/stars/truera/trulens) |
| EvalPlus | Rigorous evaluation framework for LLM4Code | [Github](https://github.com/evalplus/evalplus) | ![GitHub stars](https://img.shields.io/github/stars/evalplus/evalplus) |
| LightEval | All-in-one toolkit for evaluating LLMs | [Github](https://github.com/huggingface/lighteval) | ![GitHub stars](https://img.shields.io/github/stars/huggingface/lighteval) |
| LangTest | Test suite for comparing LLM models on accuracy, bias, fairness and robustness | [Github](https://github.com/JohnSnowLabs/langtest) | ![GitHub stars](https://img.shields.io/github/stars/JohnSnowLabs/langtest) |
| AgentEvals | Evaluators and utilities for measuring agent performance | [Github](https://github.com/langchain-ai/agentevals) | ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/agentevals) |