---
layout: page
title: Neural Price Forecasting System
description: Multi-Model Time Series Prediction with Automated Hyperparameter Tuning
img: assets/img/quantops-forecast.png
importance: 1
category: "QuantOps"
---

A machine learning application for forecasting commodity and financial asset prices using state-of-the-art neural time series models with automated hyperparameter optimization. The system supports both univariate and multivariate forecasting with integrated technical analysis and calendar feature engineering.

## Architecture
- Modular ML pipeline with separation of data ingestion, transformation, and prediction
- Automated experiment tracking with model versioning
- PyTorch Lightning for distributed training orchestration
- Optuna TPE sampler for hyperparameter optimization

## Key Features
**Neural Forecasting Models**: AutoNHITS with hierarchical interpolation, AutoTSMixerx with exogenous variable support, AutoiTransformer for multivariate analysis, and AutoTSMixer for temporal pattern recognition.

**Automated Optimization**: Hyperparameter search with configurable search spaces, 10 trials per model, early stopping based on validation loss (patience: 5), and MAE loss function for robust predictions.

**Feature Engineering**: Calendar features (day_of_week, is_weekend, month, quarter, year, is_holiday), standard scaling for univariate data, robust scaling for multivariate data, and automatic train/validation/test splits.

**Technical Analysis Integration**: Moving averages, RSI with configurable periods, Bollinger Bands, MACD with signal line, Ultimate Oscillator, and ADX with directional movement indicators using pandas_ta.

**Training Infrastructure**: PyTorch Lightning with CSV logging, maximum 1000 training steps, validation checks every 50 steps, and organized artifact storage with model checkpoints.

## Technical Stack
- NeuralForecast for time series models
- Optuna for hyperparameter optimization
- PyTorch Lightning for training orchestration
- pandas_ta for technical analysis indicators
- scikit-learn for preprocessing pipelines

## Use Cases
- Crude oil price forecasting (Brent futures)
- Multi-asset price prediction with exogenous variables
- Integration of Gas prices, DXY index, and Brent futures data
