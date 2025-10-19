---
layout: page
title: CFO Copilot
description: AI-Powered Personal Finance Management Platform
img: assets/img/cfo-copilot.png
importance: 1
category: "Financial Technology"
---

An AI-powered financial advisory platform using multi-agent orchestration to analyze financial documents and generate actionable insights. The system processes natural language queries, executes dynamic Python code for analysis, and delivers HTML infographics and React-compatible Plotly charts through conversational interfaces.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cfo-copilot.png" title="Multi-Agent AI Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Multi-agent AI orchestration with semantic processing and dynamic analysis capabilities.
</div>

## Technical Stack
- **Backend**: FastAPI with async/await, Cloud SQL PostgreSQL
- **AI/ML**: OpenAI GPT-4o, LangGraph for multi-agent workflows
- **Vector Database**: ChromaDB with text-embedding-3-large (1024 dimensions)
- **Frontend**: React 18 with TypeScript, Vite, Tailwind CSS
- **Visualization**: Plotly for interactive charts
- **Infrastructure**: Google Cloud Run, Cloud Storage, Vercel

## Multi-Agent Architecture
**LangGraph Workflow**: Five specialized agents orchestrate query processing:
- Query Analyzer: Intent classification and routing
- Data Retriever: Semantic search across financial documents
- Code Generator: Dynamic Python code creation for analysis
- Code Executor: Sandboxed execution environment
- Response Formatter: Output formatting with visualizations

**Smart Caching**: Similarity-based response caching with 10-minute TTL and 85% threshold, achieving 65% cache hit rate

## Key Features
**Natural Language Processing**: Conversational interface for financial queries with intent classification and entity extraction

**Document Processing**: Multi-format ingestion (PDF, Excel, CSV, JSON) with OCR fallback for scanned documents

**Dynamic Code Execution**: Secure sandboxed Python environment (pandas, numpy, plotly) with AST validation and memory/time limits

**Interactive Visualizations**: Generates HTML infographics and React-compatible Plotly charts embedded directly in conversational responses

**Subscription Management**: AI-powered recognition of subscription renewals with lifecycle monitoring and cost analysis

## Performance Metrics
- <1 seconds for similarity-based cached queries, <30 seconds for new deep analyses
- 50+ simultaneous users supported with horizontal scaling
- 95%+ code execution and user query success rate