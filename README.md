# RAGtoolkit 

<div align="center">
  <p>
    <a href="https://www.linkedin.com/in/long-le-713b41111/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a> |
    <a href="https://github.com/lehoanglong95" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" /></a>
  </p>
</div>

## Introduction

RAGtoolkit is a comprehensive repository that serves as a guide to building Retrieval-Augmented Generation (RAG) applications. It provides a curated collection of tools, libraries, and frameworks necessary for developing robust RAG systems. This toolkit aims to simplify the process of implementing RAG by offering detailed explanations of key components and recommendations for the most effective libraries for each aspect of the RAG pipeline.

## What is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that enhances Large Language Models (LLMs) by supplementing the generation process with relevant information retrieved from external knowledge sources. This approach addresses the limitations of traditional LLMs by:

1. Providing access to information beyond the model's training data
2. Reducing hallucinations by grounding responses in factual sources
3. Enabling real-time knowledge updates without retraining the model
4. Improving transparency through citation of sources

The RAG process typically involves the following steps:
1. Ingesting and preprocessing documents
2. Chunking documents into manageable segments
3. Creating vector embeddings of these chunks
4. Storing documents and vectors in appropriate databases
5. Processing user queries through semantic and/or keyword search
6. Retrieving relevant documents
7. Ranking or reranking the retrieved documents
8. Using an LLM to generate a response based on the retrieved information

### RAG Architecture Diagram

![RAG Architecture](RAG%20Diagram.png)

## RAG Components

| Component | Description |
|-----------|-------------|
| [<img src="https://img.shields.io/badge/-Document%20Ingestor-orange?style=flat-square&logo=database" alt="Document Ingestor" /> Document Ingestor](#document-ingestor) | Tools for ingesting and processing raw documents. Document loaders, parsers, and preprocessing tools |
| [<img src="https://img.shields.io/badge/-RAG%20Framework-blue?style=flat-square&logo=framework" alt="RAG Framework" /> RAG Framework](#rag-framework) | End-to-end frameworks for building RAG applications. Unified solutions for RAG implementation |
| [<img src="https://img.shields.io/badge/-Vector%20Database-green?style=flat-square&logo=database" alt="Vector Database" /> Vector Database](#vector-database) | Databases optimized for storing and searching vector embeddings. Vector storage, similarity search, and indexing |
| [<img src="https://img.shields.io/badge/-Document%20Database-yellow?style=flat-square&logo=mongodb" alt="Document Database" /> Document Database](#document-database) | Databases for storing and retrieving text documents. Storage for raw and processed documents |
| [<img src="https://img.shields.io/badge/-LLM-purple?style=flat-square&logo=openai" alt="LLM" /> LLM](#llm) | Large Language Models for generating responses. LLM providers and frameworks |
| [<img src="https://img.shields.io/badge/-Embedding-teal?style=flat-square&logo=tensorflow" alt="Embedding" /> Embedding](#embedding) | Models and services for creating text embeddings. Embedding models and APIs |
| [<img src="https://img.shields.io/badge/-LLM%20Observability-red?style=flat-square&logo=grafana" alt="LLM Observability" /> LLM Observability](#llm-observability) | Tools for monitoring and analyzing LLM performance. Logging, tracing, and analytics |
| [<img src="https://img.shields.io/badge/-Prompt%20Techniques-pink?style=flat-square&logo=textpattern" alt="Prompt Techniques" /> Prompt Techniques](#prompt-techniques) | Methods for effective prompt engineering. Prompt templates and frameworks |
| [<img src="https://img.shields.io/badge/-Evaluation-brown?style=flat-square&logo=checklist" alt="Evaluation" /> Evaluation](#evaluation) | Tools for assessing RAG system performance. Metrics and evaluation frameworks |

## Document Ingestor

Tools and libraries for ingesting various document formats, extracting text, and preparing data for further processing.

| Library | Description | Link |
|---------|-------------|------|
| LangChain Document Loaders | Comprehensive set of document loaders for various file types | [GitHub](https://github.com/langchain-ai/langchain) |
| Unstructured | Library for pre-processing and extracting content from raw documents | [GitHub](https://github.com/Unstructured-IO/unstructured) |
| Haystack DocumentStore | Flexible document processing and storage | [GitHub](https://github.com/deepset-ai/haystack) |
| PyPDF | Library for reading and manipulating PDF files | [GitHub](https://github.com/py-pdf/pypdf) |
| BeautifulSoup | Library for web scraping and HTML parsing | [GitHub](https://github.com/wention/BeautifulSoup4) |

## RAG Framework

End-to-end frameworks that provide integrated solutions for building RAG applications.

| Framework | Description | Link |
|-----------|-------------|------|
| LangChain | Framework for building applications with LLMs and integrating with various data sources | [GitHub](https://github.com/langchain-ai/langchain) |
| LlamaIndex | Data framework for building RAG systems with structured data | [GitHub](https://github.com/jerryjliu/llama_index) |
| Haystack | End-to-end framework for building NLP pipelines | [GitHub](https://github.com/deepset-ai/haystack) |
| RAGAS | Evaluation framework for RAG systems | [GitHub](https://github.com/explodinggradients/ragas) |
| DSPy | Programming framework for leveraging LLMs with retrieval | [GitHub](https://github.com/stanfordnlp/dspy) |

## Vector Database

Databases optimized for storing and efficiently searching vector embeddings.

| Database | Description | Link |
|----------|-------------|------|
| Pinecone | Managed vector database for semantic search | [Website](https://www.pinecone.io/) |
| Weaviate | Open-source vector search engine | [GitHub](https://github.com/weaviate/weaviate) |
| Milvus | Open-source vector database | [GitHub](https://github.com/milvus-io/milvus) |
| Qdrant | Vector similarity search engine | [GitHub](https://github.com/qdrant/qdrant) |
| Chroma | Open-source embedding database designed for RAG applications | [GitHub](https://github.com/chroma-core/chroma) |
| FAISS | Efficient similarity search library from Facebook AI Research | [GitHub](https://github.com/facebookresearch/faiss) |

## Document Database

Databases designed for storing and retrieving text documents.

| Database | Description | Link |
|----------|-------------|------|
| MongoDB | General-purpose document database | [Website](https://www.mongodb.com/) |
| Elasticsearch | Search and analytics engine that can store documents | [Website](https://www.elastic.co/) |
| PostgreSQL | Relational database with JSON support | [Website](https://www.postgresql.org/) |
| Redis | In-memory data structure store with document capabilities | [Website](https://redis.io/) |
| DynamoDB | NoSQL document database service by AWS | [Website](https://aws.amazon.com/dynamodb/) |

## LLM

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

| Tool | Description | Link |
|------|-------------|------|
| LangSmith | Debugging, testing, and monitoring platform for LLM applications | [Website](https://www.langchain.com/langsmith) |
| Weights & Biases | MLOps platform with LLM tracking | [Website](https://wandb.ai/site) |
| Phoenix | Open-source observability for LLM applications | [GitHub](https://github.com/Arize-ai/phoenix) |
| DeepChecks | Validation and testing for LLMs | [Website](https://deepchecks.com/) |
| Helicone | LLM observability and analytics platform | [Website](https://www.helicone.ai/) |
| LiteLLM | Standardized API for LLM providers with observability | [GitHub](https://github.com/BerriAI/litellm) |

## Prompt Techniques

Methods and frameworks for effective prompt engineering in RAG systems.

| Technique/Library | Description | Link |
|-------------------|-------------|------|
| LangChain Prompts | Templates and composition tools for prompts | [Documentation](https://python.langchain.com/docs/modules/model_io/prompts/) |
| Guidance | Language for controlling LLMs | [GitHub](https://github.com/guidance-ai/guidance) |
| PROMPTIFY | Create, test and deploy prompts | [GitHub](https://github.com/promptslab/promptify) |
| PromptPerfect | Tool for optimizing prompts | [Website](https://promptperfect.jina.ai/) |
| DynaPrompt | Dynamic prompt generation | [GitHub](https://github.com/xtheai/dynaprompt) |
| Prompt Engineering Guide | Comprehensive guide to prompt engineering | [GitHub](https://github.com/dair-ai/Prompt-Engineering-Guide) |

## Evaluation

Tools and frameworks for assessing and improving RAG system performance.

| Tool | Description | Link |
|------|-------------|------|
| RAGAS | Evaluation framework specifically for RAG systems | [GitHub](https://github.com/explodinggradients/ragas) |
| TruLens | Open-source package for LLM evaluation with RAG-specific metrics | [GitHub](https://github.com/truera/trulens) |
| DeepEval | Evaluation library for LLM applications | [GitHub](https://github.com/confident-ai/deepeval) |
| OpenAI Evals | Framework for evaluating LLMs | [GitHub](https://github.com/openai/evals) |
| LangSmith Evaluators | Evaluation tools integrated with LangChain | [Documentation](https://docs.smith.langchain.com/evaluation) |
| Promptfoo | Open-source tool for testing and evaluating prompts | [GitHub](https://github.com/promptfoo/promptfoo) | 