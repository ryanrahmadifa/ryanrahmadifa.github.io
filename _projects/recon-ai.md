---
layout: page
title: Recon-AI
description: Enterprise-Grade Bank Reconciliation Engine with Tri-Layer AI Processing
img: assets/img/recon-ai.png
importance: 1
category: "FinOps"
---

A financial reconciliation platform combining mathematical algorithms with AI to match bank transactions with ledger entries. The system uses a tri-layer approach: proprietary weighted scoring for initial matching, followed by GPT-4 for complex cases requiring contextual understanding, and a conflict resolution layer for single vs. split payment scenarios.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/recon-ai.png" title="High-Performance Computing Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Tri-layer processing with mathematical scoring, AI-powered decision making, and conflict resolution.
</div>

## Technical Stack
- **Backend**: FastAPI with async/await patterns, async AI API calls
- **Vector Store**: ChromaDB with text-embedding-3-large (1024 dimensions)
- **Database**: PostgreSQL Cloud SQL (production), SQLite (development)
- **Cache**: Async SQLite with WAL mode, 10-minute TTL
- **Infrastructure**: Google Cloud Run with auto-scaling

## Key Features
**Split Payment Detection**: Many-to-many matching for complex scenarios (one bank transaction to multiple ledger entries, or vice versa)

**Intelligent Batching**: Concurrent processing with controlled parallelism, reducing LLM API costs by 80% while maintaining accuracy

**Conflict Resolution**: LLM-arbitrated matching with evidence-based decisions and complete audit trails for compliance

**Performance Optimization**: Memory-efficient processing handles 100k+ transactions with parallel execution and multi-level caching

## Performance Metrics
- 95% overall reconciliation accuracy
- 80% reduction in LLM API costs through intelligent batching
- 1,000 transactions/minute processing speed
- 100+ concurrent reconciliation jobs supported
- Sub-512MB memory usage for typical scenarios