---
layout: page
title: Market Sentiment Analysis System
description: Transformer-Based Financial News Classification with MLOps Pipeline
img: assets/img/quantops-trend.png
importance: 1
category: "QuantOps"
---

A full-stack NLP application that classifies market sentiment and categorizes financial news using DistilRoBERTa transformer models. The system processes news headlines to predict sentiment categories (Bearish, Neutral, Bullish) with confidence scores, enabling data-driven trading decisions through real-time semantic analysis.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/quantops-trend.png" title="Market Semantic Analysis Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results of semantic analysis on news regarding geopolitical risk, oil, and gas.
</div>

## Architecture
- **Backend**: Spring Boot (Java) with Gradle for REST API services
- **Frontend**: Next.js with TypeScript and Tailwind CSS
- **ML Service**: Python-based prediction pipeline with GPU acceleration
- **MLOps**: MLflow for experiment tracking and model versioning

## Key Features
**Semantic Analysis Engine**: DistilRoBERTa transformer with sigmoid normalization for three-class sentiment classification, utilizing RobertaTokenizerFast with 512 max token length and attention mask generation for improved accuracy.

**News Categorization**: Automated multi-class topic classification system with integrated verification workflow for organizing headlines by subject matter.

**MLOps Pipeline**: Complete experiment tracking with MLflow, automated cross-validation, hyperparameter optimization using Optuna TPE sampler, and model checkpoint management for reproducibility.

**Time Series Forecasting**: Integration of TSMixer, TSMixerx, NBEATSx, and NHITS models for advanced market prediction capabilities.

## Technical Stack
- Transformers (Hugging Face) with PyTorch and CUDA support
- NeuralForecast for time series modeling
- MLflow for experiment tracking and artifact management
- Pandas for data manipulation and batch processing
- Spring Boot backend with Next.js frontend

## Performance
- GPU-accelerated inference with CUDA support
- Batch prediction pipeline supporting DataFrame operations
- Confidence score outputs for each sentiment category
- Automated model versioning and artifact logging
