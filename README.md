# RAG All-in-one

## Hello there! 👋

I'm Long Le, a Machine Learning Engineer passionate about building AI systems. This repository is my collection of RAG (Retrieval-Augmented Generation) resources to help you build powerful AI applications.

Feel free to connect with me on social media to discuss AI, machine learning, or this project:

<div align="center">
  <p>
    <a href="https://www.linkedin.com/in/long-le-713b41111/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a> |
    <a href="https://github.com/lehoanglong95" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" /></a>
  </p>
</div>

## Introduction

RAG All-in-one is a guide to building Retrieval-Augmented Generation (RAG) applications. It offers a comprehensive collection of tools, libraries, and frameworks for RAG systems, organized by key components of the RAG architecture. This resource serves as a centralized directory to help you discover the most relevant technologies for each part of your RAG pipeline.

### RAG Architecture Diagram

![RAG Architecture](RAG%20Diagram.png)

## RAG Components

| Component | Description |
|-----------|-------------|
| [📚 Courses and Learning Materials](#courses-and-learning-materials) | Comprehensive courses and learning resources for mastering RAG systems |
| [📄 Document Ingestor](#document-ingestor) | Tools for ingesting and processing raw documents. Document loaders, parsers, and preprocessing tools |
| [✂️ Chunking Techniques](#chunking-techniques) | Methods and tools for breaking down documents into manageable pieces for processing and retrieval |
| [🤖 Agent Framework](#agent-framework) | End-to-end frameworks for building RAG applications. Unified solutions for RAG implementation |
| [📀 Database](#database) | Databases optimized for storing and searching vector embeddings. Vector storage, similarity search, and indexing |
| [💻 LLM](#llm) | Large Language Models for generating responses. LLM providers and frameworks |
| [📝 Embedding](#embedding) | Models and services for creating text embeddings. Embedding models and APIs |
| [🖥️ LLM Observability](#llm-observability) | Tools for monitoring and analyzing LLM performance. Logging, tracing, and analytics |
| [📕 Prompt Techniques](#prompt-techniques) | Methods for effective prompt engineering. Prompt templates and frameworks |
| [🤔 Evaluation](#evaluation) | Tools for assessing RAG system performance. Metrics and evaluation frameworks |
| [📺 User Interface](#user-interface) | Tools for building interactive AI interfaces. UI frameworks for RAG applications |
| [🚀 Complete RAG Applications](#complete-rag-applications) | Ready-to-use, comprehensive RAG systems that integrate various components of the RAG stack |

## Courses and Learning Materials

Comprehensive courses and learning resources for mastering RAG systems.

| Course Name | Platform | Description | Link | Level |
|-------------|----------|-------------|------|-------|
| Building and Evaluating Advanced RAG Applications | ![DeepLearning.AI](https://img.shields.io/badge/DeepLearning.AI-FF6B00?style=flat-square&logo=deeplearning.ai&logoColor=white) | Advanced retrieval methods, sentence-window retrieval, auto-merging retrieval, and evaluation metrics | [Link](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) | Beginner |
| Learn RAG with LLMWare | ![Udemy](https://img.shields.io/badge/Udemy-EC5252?style=flat-square&logo=udemy&logoColor=white) | Fundamentals of RAG, parsing, embeddings, prompting, and semantic querying | [Link](https://www.udemy.com/course/learn-rag-with-llmware-2024/) | Beginner to Intermediate |
| Learn AI Skills: RAG Course | ![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=flat-square&logo=youtube&logoColor=white) | Basics to end-to-end RAG system creation with code examples | [Link](https://www.youtube.com/watch?v=cYRcdsqFAmY&t=5s) | Beginner to Advanced |
| Building Multimodal Search and RAG | ![DeepLearning.AI](https://img.shields.io/badge/DeepLearning.AI-FF6B00?style=flat-square&logo=deeplearning.ai&logoColor=white) | Multimodal RAG systems using contrastive learning for images, audio, video alongside text data | [Link](https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/) | Intermediate |
| AI Enhancement with Knowledge Graphs - Mastering RAG Systems | ![Coursera](https://img.shields.io/badge/Coursera-0056D2?style=flat-square&logo=coursera&logoColor=white) | Integrating Knowledge Graphs with RAG systems for improved contextual understanding | [Link](https://www.coursera.org/learn/packt-ai-enhancement-with-knowledge-graphs-mastering-rag-systems-lnmqm) | Intermediate |
| Introduction to RAG | ![Coursera](https://img.shields.io/badge/Coursera-0056D2?style=flat-square&logo=coursera&logoColor=white) | Building an end-to-end RAG system with Pandas, SentenceTransformers, Qdrant, and LLMs | [Link](https://www.coursera.org/projects/introduction-to-rag) | Intermediate |
| Retrieval Augmented Generation (RAG) with LangChain | ![DataCamp](https://img.shields.io/badge/DataCamp-03EF62?style=flat-square&logo=datacamp&logoColor=white) | Using LangChain for integrating external data into LLMs, text splitting, embeddings | [Link](https://www.datacamp.com/courses/retrieval-augmented-generation-rag-with-langchain) | Intermediate |
| Retrieval Augmented Generation (RAG) for Developers | ![Pluralsight](https://img.shields.io/badge/Pluralsight-F15B2D?style=flat-square&logo=pluralsight&logoColor=white) | Modular RAGs, optimizing retrieval techniques, relevance ranking | [Link](https://www.pluralsight.com/paths/retrieval-augmented-generation-rag-for-developers) | Intermediate |
| Retrieval Augmented Generation LlamaIndex & LangChain Course | ![Activeloop](https://img.shields.io/badge/Activeloop-1E88E5?style=flat-square&logo=activeloop&logoColor=white) | Production-oriented RAG systems using LlamaIndex and Deep Lake | [Link](https://learn.activeloop.ai/courses/rag) | Intermediate to Advanced |
| The RAG Bootcamp | ![Zero to Mastery](https://img.shields.io/badge/Zero_to_Mastery-1E88E5?style=flat-square&logo=zerotomastery&logoColor=white) | Retrieval systems, multimodal RAG applications, and hands-on projects | [Link](https://zerotomastery.io/courses/ai-engineer-bootcamp-retrieval-augmented-generation/) | Intermediate to Advanced |
| AI: Advanced RAG | ![edX](https://img.shields.io/badge/edX-02262B?style=flat-square&logo=edx&logoColor=white) | Enterprise-grade RAG techniques, embedding strategies, document processing | [Link](https://www.edx.org/learn/computer-science/pragmatic-ai-labs-advanced-rag?index=product&queryID=67699afa89a44a42e3aec1413f0152ed&position=1&results_level=first-level-results&term=AI%3A+Advanced+RAG%09&objectID=course-69f4b8db-8a82-4439-87a2-7f5b1fe5334b&campaign=Advanced+RAG) | Advanced |
| Advanced Retrieval-Augmented Generation (RAG) for Large Language Models | ![FutureLearn](https://img.shields.io/badge/FutureLearn-DE00A5?style=flat-square&logo=futurelearn&logoColor=white) | Advanced embedding strategies, hybrid search systems, sparse indexing | [Link](https://www.futurelearn.com/courses/rag-for-large-language-models) | Advanced |

## Document Ingestor

Tools and libraries for ingesting various document formats, extracting text, and preparing data for further processing.

| Library | Description | Link | GitHub Stars 🌟 |
|---------|-------------|------|-------------|
| LangChain Document Loaders | Comprehensive set of document loaders for various file types | [GitHub](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/document_loaders) | ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/langchain) |
| LlamaIndex Reader | Flexible document parsing and chunking capabilities for various file formats | [GitHub](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers) | ![GitHub stars](https://img.shields.io/github/stars/jerryjliu/llama_index) |
| Docling | Document processing tool that parses diverse formats with advanced PDF understanding and AI integrations | [GitHub](https://github.com/docling-project/docling) | ![GitHub stars](https://img.shields.io/github/stars/docling-project/docling) |
| Unstructured | Library for pre-processing and extracting content from raw documents | [GitHub](https://github.com/Unstructured-IO/unstructured) | ![GitHub stars](https://img.shields.io/github/stars/Unstructured-IO/unstructured) |
| PyPDF | Library for reading and manipulating PDF files | [GitHub](https://github.com/py-pdf/pypdf) | ![GitHub stars](https://img.shields.io/github/stars/py-pdf/pypdf) |
| PyMuPDF | A Python binding for MuPDF, offering fast PDF processing capabilities | [GitHub](https://github.com/pymupdf/PyMuPDF) | ![GitHub stars](https://img.shields.io/github/stars/pymupdf/PyMuPDF) |
| MegaParse | Versatile parser for text, PDFs, PowerPoint, and Word documents with lossless information extraction | [GitHub](https://github.com/QuivrHQ/MegaParse) | ![GitHub stars](https://img.shields.io/github/stars/QuivrHQ/MegaParse) |
| Adobe PDF Extract | A service provided by Adobe for extracting content from PDF documents | [Link](https://developer.adobe.com/document-services/docs/overview/legacy-documentation/pdf-extract-api/quickstarts/python/) |  |
| Azure AI Document Intelligence | A service provided by Azure for extracting content including text, tables, images from PDF documents | [Link](https://developer.adobe.com/document-services/docs/overview/legacy-documentation/pdf-extract-api/quickstarts/python/) |  |

## Chunking Techniques

Methods and tools for breaking down documents into manageable pieces for processing and retrieval.

| Technique | Description | Link | Code Example |
|-----------|-------------|----------|--------------|
| Fixed Size Chunking | Splits text into chunks of specified character length. Simple and computationally efficient. Key concepts: chunk size, overlap, separator. | [Link](https://www.youtube.com/watch?v=8OJC21T2SL4&t=450s) | [Code Example](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/) |
| Recursive Chunking | Hierarchically divides text using multiple separators in sequence. Respects text structure by recursively applying different separators. | [Link](https://www.youtube.com/watch?v=8OJC21T2SL4&t=966s) | [Code Example](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/) |
| Document Based Chunking | Splits content according to document's inherent structure (headers, code blocks, tables, etc.). Format-aware chunking for Markdown, Python, JS, etc. | [Link](https://medium.com/@david.richards.tech/document-chunking-for-rag-ai-applications-04363d48fbf7) | [Code Example](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/) |
| Semantic Chunking | Creates chunks based on semantic similarity rather than size. Keeps related content together by analyzing embedding similarity at potential breakpoints. | [Link](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1933s) | [Code Example](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_parsers/semantic_chunking.ipynb) |
| Agentic Chunking | Uses LLM-based agents to intelligently determine chunk boundaries based on context and content. Can identify standalone propositions for optimal chunking. | [Link](https://www.youtube.com/watch?v=8OJC21T2SL4&t=2889s) | [Code Example](https://github.com/Ranjith-JS2803/Agentic-Chunker) |

## Agent Framework

End-to-end frameworks that provide integrated solutions for building RAG applications.

| Library | Description | Link | GitHub Stars 🌟 |
|-----------|-------------|------|-------------|
| LangChain | Framework for building applications with LLMs and integrating with various data sources | [GitHub](https://github.com/langchain-ai/langchain) | ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/langchain) |
| LlamaIndex | Data framework for building RAG systems with structured data | [GitHub](https://github.com/jerryjliu/llama_index) | ![GitHub stars](https://img.shields.io/github/stars/jerryjliu/llama_index) |
| Haystack | End-to-end framework for building NLP pipelines | [GitHub](https://github.com/deepset-ai/haystack) | ![GitHub stars](https://img.shields.io/github/stars/deepset-ai/haystack) |
| SmolAgents | A barebones library for agents | [GitHub](https://github.com/huggingface/smolagents) | ![GitHub stars](https://img.shields.io/github/stars/huggingface/smolagents) |
| txtai | Open-source embeddings database for semantic search and LLM workflows | [GitHub](https://github.com/neuml/txtai) | ![GitHub stars](https://img.shields.io/github/stars/neuml/txtai) |
| Pydantic AI | Agent Framework / shim to use Pydantic with LLMs | [GitHub](https://github.com/pydantic/pydantic-ai) | ![GitHub stars](https://img.shields.io/github/stars/pydantic/pydantic-ai) |
| OpenAI Agent | A lightweight, powerful framework for multi-agent workflows | [GitHub](https://github.com/openai/openai-agents-python) | ![GitHub stars](https://img.shields.io/github/stars/openai/openai-agents-python) |

## Database 

Databases optimized for storing and efficiently searching vector embeddings/text documents.

| Database | Description | Link | GitHub Stars 🌟 |
|----------|-------------|------|-------------|
| FAISS | Efficient similarity search library from Facebook AI Research | [GitHub](https://github.com/facebookresearch/faiss) | ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/faiss) |
| Milvus | Open-source vector database | [GitHub](https://github.com/milvus-io/milvus) | ![GitHub stars](https://img.shields.io/github/stars/milvus-io/milvus) |
| Qdrant | Vector similarity search engine | [GitHub](https://github.com/qdrant/qdrant) | ![GitHub stars](https://img.shields.io/github/stars/qdrant/qdrant) |
| Chroma | Open-source embedding database designed for RAG applications | [GitHub](https://github.com/chroma-core/chroma) | ![GitHub stars](https://img.shields.io/github/stars/chroma-core/chroma) |
| pgvector | Open-source vector similarity search for Postgres | [GitHub](https://github.com/pgvector/pgvector) | ![GitHub stars](https://img.shields.io/github/stars/pgvector/pgvector) |
| Weaviate | Open-source vector search engine | [GitHub](https://github.com/weaviate/weaviate) | ![GitHub stars](https://img.shields.io/github/stars/weaviate/weaviate) |
| LanceDB | Developer-friendly, embedded retrieval engine for multimodal AI | [GitHub](https://github.com/lancedb/lancedb) | ![GitHub stars](https://img.shields.io/github/stars/lancedb/lancedb) |
| Pinecone | Managed vector database for semantic search | [Link](https://www.pinecone.io/) |  |
| MongoDB | General-purpose document database | [Link](https://www.mongodb.com/) |  |
| Elasticsearch | Search and analytics engine that can store documents | [Link](https://www.elastic.co/) |  |

## LLM

Large Language Models and platforms for generating responses based on retrieved context.

| LLM | Description | Link |
|-----|-------------|------|
| OpenAI API | Access to GPT models through API | [Link](https://platform.openai.com/) |
| Claude | Anthropic's Claude series of LLMs | [Link](https://www.anthropic.com/claude) |
| Hugging Face LLM Models| Platform for open-source NLP models | [Link](https://huggingface.co/collections/open-llm-leaderboard/open-llm-leaderboard-best-models-652d6c7965a4619fb5c27a03) |
| LLaMA | Meta's open-source large language model | [Link](https://github.com/facebookresearch/llama) |
| Mistral | Open-source and commercial models | [Link](https://mistral.ai/) |
| Cohere | API access to generative and embedding models | [Link](https://cohere.com/) |
| DeepSeek | Advanced large language models for various applications | [Link](https://www.deepseek.com/) |
| Qwen | Alibaba Cloud's large language model accessible via API | [Link](https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api) |
| Ollama | Run open-source LLMs locally | [Link](https://github.com/ollama/ollama) |

## Embedding

Models and services for creating vector representations of text.

| Embedding Solution | Description | Link |
|-------------------|-------------|------|
| OpenAI Embeddings | API for text-embedding-ada-002 and newer models | [Link](https://platform.openai.com/docs/guides/embeddings) |
| Sentence Transformers | Python framework for state-of-the-art sentence embeddings | [Link](https://github.com/UKPLab/sentence-transformers) |
| Cohere Embed | Specialized embedding models API | [Link](https://cohere.com/embed) |
| Hugging Face Embeddings | Various embedding models | [Link](https://huggingface.co/models?pipeline_tag=feature-extraction) |
| E5 Embeddings | Microsoft's text embeddings | [Link](https://huggingface.co/intfloat/e5-large-v2) |
| BGE Embeddings | BAAI general embeddings | [Link](https://huggingface.co/BAAI/bge-large-en-v1.5) |

## LLM Observability

Tools for monitoring, analyzing, and improving LLM applications.


| Library | Description | Link | GitHub Stars 🌟                                                                       |
|------|-------------|------|---------------------------------------------------------------------------|
| Langfuse | Open source LLM engineering platform | [GitHub](https://github.com/langfuse/langfuse) | ![GitHub stars](https://img.shields.io/github/stars/langfuse/langfuse) |
| Opik/Comet | Debug, evaluate, and monitor LLM applications with tracing, evaluations, and dashboards | [GitHub](https://github.com/comet-ml/opik) | ![GitHub stars](https://img.shields.io/github/stars/comet-ml/opik) |
| Phoenix/Arize | Open-source observability for LLM applications | [GitHub](https://github.com/Arize-ai/phoenix) | ![GitHub stars](https://img.shields.io/github/stars/Arize-ai/phoenix) |
| Helicone | Open source LLM observability platform. One line of code to monitor, evaluate, and experiment | [GitHub](https://github.com/helicone/helicone) | ![GitHub stars](https://img.shields.io/github/stars/helicone/helicone) |
| Openlit | Open source platform for AI Engineering: OpenTelemetry-native LLM Observability, GPU Monitoring, Guardrails, Evaluations, Prompt Management, Vault, Playground | [GitHub](https://github.com/openlit/openlit) | ![GitHub stars](https://img.shields.io/github/stars/openlit/openlit) |
| Lunary | The production toolkit for LLMs. Observability, prompt management and evaluations. | [GitHub](https://github.com/lunary-ai/lunary) | ![GitHub stars](https://img.shields.io/github/stars/lunary-ai/lunary) |
| Langtrace | OpenTelemetry-based observability tool for LLM applications with real-time tracing and metrics | [GitHub](https://github.com/Scale3-Labs/langtrace) | ![GitHub stars](https://img.shields.io/github/stars/Scale3-Labs/langtrace) |

## Prompt Techniques

Methods and frameworks for effective prompt engineering in RAG systems.

### Open Source Prompt Engineering Tools

| Library | Description | Link | GitHub Stars 🌟 |
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

| Library | Description | Link | GitHub Stars 🌟 |
|------|-------------|------|-------|
| FastChat | Open platform for training, serving, and evaluating LLM-based chatbots | [GitHub](https://github.com/lm-sys/fastchat) | ![GitHub stars](https://img.shields.io/github/stars/lm-sys/fastchat) |
| OpenAI Evals | Framework for evaluating LLMs and LLM systems | [GitHub](https://github.com/openai/evals) | ![GitHub stars](https://img.shields.io/github/stars/openai/evals) |
| RAGAS | Ultimate toolkit for evaluating and optimizing RAG systems | [GitHub](https://github.com/explodinggradients/ragas) | ![GitHub stars](https://img.shields.io/github/stars/explodinggradients/ragas) |
| Promptfoo | Open-source tool for testing and evaluating prompts | [GitHub](https://github.com/promptfoo/promptfoo) | ![GitHub stars](https://img.shields.io/github/stars/promptfoo/promptfoo) |
| DeepEval | Comprehensive evaluation library for LLM applications | [GitHub](https://github.com/confident-ai/deepeval) | ![GitHub stars](https://img.shields.io/github/stars/confident-ai/deepeval) |
| Giskard | Open-source evaluation and testing for ML & LLM systems | [GitHub](https://github.com/giskard-ai/giskard) | ![GitHub stars](https://img.shields.io/github/stars/giskard-ai/giskard) |
| PromptBench | Unified evaluation framework for large language models | [GitHub](https://github.com/microsoft/promptbench) | ![GitHub stars](https://img.shields.io/github/stars/microsoft/promptbench) |
| TruLens | Evaluation and tracking for LLM experiments with RAG-specific metrics | [GitHub](https://github.com/truera/trulens) | ![GitHub stars](https://img.shields.io/github/stars/truera/trulens) |
| EvalPlus | Rigorous evaluation framework for LLM4Code | [GitHub](https://github.com/evalplus/evalplus) | ![GitHub stars](https://img.shields.io/github/stars/evalplus/evalplus) |
| LightEval | All-in-one toolkit for evaluating LLMs | [GitHub](https://github.com/huggingface/lighteval) | ![GitHub stars](https://img.shields.io/github/stars/huggingface/lighteval) |
| LangTest | Test suite for comparing LLM models on accuracy, bias, fairness and robustness | [GitHub](https://github.com/JohnSnowLabs/langtest) | ![GitHub stars](https://img.shields.io/github/stars/JohnSnowLabs/langtest) |
| AgentEvals | Evaluators and utilities for measuring agent performance | [GitHub](https://github.com/langchain-ai/agentevals) | ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/agentevals) |

## User Interface

Tools and frameworks for building interactive user interfaces for RAG applications.

| Library | Description | Link | GitHub Stars 🌟 |
|------|-------------|------|-------|
| Streamlit | Turn data scripts into shareable web apps in minutes | [GitHub](https://github.com/streamlit/streamlit) | ![GitHub stars](https://img.shields.io/github/stars/streamlit/streamlit) |
| Gradio | Build and share user interfaces for machine learning models | [GitHub](https://github.com/gradio-app/gradio) | ![GitHub stars](https://img.shields.io/github/stars/gradio-app/gradio) |
| Chainlit | Build Python LLM apps with minimal effort | [GitHub](https://github.com/Chainlit/chainlit) | ![GitHub stars](https://img.shields.io/github/stars/Chainlit/chainlit) |
| SimpleAIChat | Lightweight Python package for creating AI chat interfaces | [GitHub](https://github.com/minimaxir/simpleaichat) | ![GitHub stars](https://img.shields.io/github/stars/minimaxir/simpleaichat) |

## Complete RAG Applications

Ready-to-use, comprehensive RAG applications that integrate various components of the RAG stack.

| Application | Description | Link | GitHub Stars 🌟 |
|-------------|-------------|------|-------------|
| RAGFlow | Open-source RAG engine based on deep document understanding for truthful question-answering with citations | [GitHub](https://github.com/infiniflow/ragflow) | ![GitHub stars](https://img.shields.io/github/stars/infiniflow/ragflow) |
| AnythingLLM | All-in-one Desktop & Docker AI application with built-in RAG, AI agents, and a no-code agent builder | [GitHub](https://github.com/Mintplex-Labs/anything-llm) | ![GitHub stars](https://img.shields.io/github/stars/Mintplex-Labs/anything-llm) |
| Kotaemon | Clean & customizable RAG UI for chatting with documents, built for both end users and developers | [GitHub](https://github.com/Cinnamon/kotaemon) | ![GitHub stars](https://img.shields.io/github/stars/Cinnamon/kotaemon) |
| Verba | Fully-customizable personal assistant utilizing RAG for querying and interacting with your data, powered by Weaviate | [GitHub](https://github.com/weaviate/Verba) | ![GitHub stars](https://img.shields.io/github/stars/weaviate/Verba) |
