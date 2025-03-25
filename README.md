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
| [📄 Document Ingestor](#document-ingestor) | Tools for ingesting and processing raw documents. Document loaders, parsers, and preprocessing tools |
| [🤖 RAG Framework](#rag-framework) | End-to-end frameworks for building RAG applications. Unified solutions for RAG implementation |
| [📀 Vector Database](#vector-database) | Databases optimized for storing and searching vector embeddings. Vector storage, similarity search, and indexing |
| [📚 Document Database](#document-database) | Databases for storing and retrieving text documents. Storage for raw and processed documents |
| [💻 LLM](#llm) | Large Language Models for generating responses. LLM providers and frameworks |
| [📝 Embedding](#embedding) | Models and services for creating text embeddings. Embedding models and APIs |
| [🖥️ LLM Observability](#llm-observability) | Tools for monitoring and analyzing LLM performance. Logging, tracing, and analytics |
| [📕 Prompt Techniques](#prompt-techniques) | Methods for effective prompt engineering. Prompt templates and frameworks |
| [🤔 Evaluation](#evaluation) | Tools for assessing RAG system performance. Metrics and evaluation frameworks |

## 📄 Document Ingestor ✅

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

## 🤖 RAG Framework

End-to-end frameworks that provide integrated solutions for building RAG applications.

| Framework | Description | Link |
|-----------|-------------|------|
| LangChain | Framework for building applications with LLMs and integrating with various data sources | [GitHub](https://github.com/langchain-ai/langchain) |
| LlamaIndex | Data framework for building RAG systems with structured data | [GitHub](https://github.com/jerryjliu/llama_index) |
| Haystack | End-to-end framework for building NLP pipelines | [GitHub](https://github.com/deepset-ai/haystack) |
| RAGAS | Evaluation framework for RAG systems | [GitHub](https://github.com/explodinggradients/ragas) |
| DSPy | Programming framework for leveraging LLMs with retrieval | [GitHub](https://github.com/stanfordnlp/dspy) |

## 📀 Vector Database 

Databases optimized for storing and efficiently searching vector embeddings.

| Database | Description | Link |
|----------|-------------|------|
| Pinecone | Managed vector database for semantic search | [Website](https://www.pinecone.io/) |
| Weaviate | Open-source vector search engine | [GitHub](https://github.com/weaviate/weaviate) |
| Milvus | Open-source vector database | [GitHub](https://github.com/milvus-io/milvus) |
| Qdrant | Vector similarity search engine | [GitHub](https://github.com/qdrant/qdrant) |
| Chroma | Open-source embedding database designed for RAG applications | [GitHub](https://github.com/chroma-core/chroma) |
| FAISS | Efficient similarity search library from Facebook AI Research | [GitHub](https://github.com/facebookresearch/faiss) |

## 📚 Document Database

Databases designed for storing and retrieving text documents.

| Database | Description | Link |
|----------|-------------|------|
| MongoDB | General-purpose document database | [Website](https://www.mongodb.com/) |
| Elasticsearch | Search and analytics engine that can store documents | [Website](https://www.elastic.co/) |
| LanceDB | Relational database with JSON support | [Website](https://lancedb.com/) |

## 💻 LLM

Large Language Models and platforms for generating responses based on retrieved context.

| LLM | Description | Link |
|-----|-------------|------|
| OpenAI API | Access to GPT models through API | [Website](https://platform.openai.com/) |
| Claude | Anthropic's Claude series of LLMs | [Website](https://www.anthropic.com/claude) |
| Hugging Face | Platform for open-source NLP models | [Website](https://huggingface.co/) |
| LLaMA | Meta's open-source large language model | [GitHub](https://github.com/facebookresearch/llama) |
| Mistral | Open-source and commercial models | [Website](https://mistral.ai/) |
| Cohere | API access to generative and embedding models | [Website](https://cohere.com/) |
| Ollama | Run open-source LLMs locally | [GitHub](https://github.com/ollama/ollama) |

## 📝 Embedding

Models and services for creating vector representations of text.

| Embedding Solution | Description | Link |
|-------------------|-------------|------|
| OpenAI Embeddings | API for text-embedding-ada-002 and newer models | [Documentation](https://platform.openai.com/docs/guides/embeddings) |
| Sentence Transformers | Python framework for state-of-the-art sentence embeddings | [GitHub](https://github.com/UKPLab/sentence-transformers) |
| Cohere Embed | Specialized embedding models API | [Website](https://cohere.com/embed) |
| Hugging Face Embeddings | Various embedding models | [Hugging Face](https://huggingface.co/models?pipeline_tag=feature-extraction) |
| E5 Embeddings | Microsoft's text embeddings | [Hugging Face](https://huggingface.co/intfloat/e5-large-v2) |
| BGE Embeddings | BAAI general embeddings | [Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5) |

## 🖥️ LLM Observability

Tools for monitoring, analyzing, and improving LLM applications.


| Tool | Description | Link | 🌟 |
|------|-------------|------|-------|
| Langfuse | Open source LLM engineering platform | [GitHub](https://github.com/langfuse/langfuse) | ![GitHub stars](https://img.shields.io/github/stars/langfuse/langfuse) |
| Helicone | Open source LLM observability platform. One line of code to monitor, evaluate, and experiment | [GitHub](https://github.com/helicone/helicone) | ![GitHub stars](https://img.shields.io/github/stars/helicone/helicone) |
| Opik/Comet | Debug, evaluate, and monitor LLM applications with tracing, evaluations, and dashboards | [GitHub](https://github.com/comet-ml/opik) | ![GitHub stars](https://img.shields.io/github/stars/comet-ml/opik) |
| Phoenix/Arize | Open-source observability for LLM applications | [GitHub](https://github.com/Arize-ai/phoenix) | ![GitHub stars](https://img.shields.io/github/stars/Arize-ai/phoenix) |
| Lunary | The production toolkit for LLMs. Observability, prompt management and evaluations. | [GitHub](https://github.com/lunary-ai/lunary) | ![GitHub stars](https://img.shields.io/github/stars/lunary-ai/lunary) |
| Openlit | Open source platform for AI Engineering: OpenTelemetry-native LLM Observability, GPU Monitoring, Guardrails, Evaluations, Prompt Management, Vault, Playground | [GitHub](https://github.com/openlit/openlit) | ![GitHub stars](https://img.shields.io/github/stars/openlit/openlit) |
| Langtrace | OpenTelemetry-based observability tool for LLM applications with real-time tracing and metrics | [GitHub](https://github.com/Scale3-Labs/langtrace) | ![GitHub stars](https://img.shields.io/github/stars/Scale3-Labs/langtrace) |

## 📕 Prompt Techniques

Methods and frameworks for effective prompt engineering in RAG systems.

| Technique/Library | Description | Link |
|-------------------|-------------|------|
| LangChain Prompts | Templates and composition tools for prompts | [Documentation](https://python.langchain.com/docs/modules/model_io/prompts/) |
| Guidance | Language for controlling LLMs | [GitHub](https://github.com/guidance-ai/guidance) |
| PROMPTIFY | Create, test and deploy prompts | [GitHub](https://github.com/promptslab/promptify) |
| PromptPerfect | Tool for optimizing prompts | [Website](https://promptperfect.jina.ai/) |
| DynaPrompt | Dynamic prompt generation | [GitHub](https://github.com/xtheai/dynaprompt) |
| Prompt Engineering Guide | Comprehensive guide to prompt engineering | [GitHub](https://github.com/dair-ai/Prompt-Engineering-Guide) |

## 🤔 Evaluation

Tools and frameworks for assessing and improving RAG system performance.

| Tool | Description | Link |
|------|-------------|------|
| RAGAS | Evaluation framework specifically for RAG systems | [GitHub](https://github.com/explodinggradients/ragas) |
| TruLens | Open-source package for LLM evaluation with RAG-specific metrics | [GitHub](https://github.com/truera/trulens) |
| DeepEval | Evaluation library for LLM applications | [GitHub](https://github.com/confident-ai/deepeval) |
| OpenAI Evals | Framework for evaluating LLMs | [GitHub](https://github.com/openai/evals) |
| LangSmith Evaluators | Evaluation tools integrated with LangChain | [Documentation](https://docs.smith.langchain.com/evaluation) |
| Promptfoo | Open-source tool for testing and evaluating prompts | [GitHub](https://github.com/promptfoo/promptfoo) | 