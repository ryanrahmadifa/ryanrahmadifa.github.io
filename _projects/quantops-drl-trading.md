---
layout: page
title: Deep RL Trading System
description: Multi-Modal DDQN Agent with Semantic Analysis and Technical Indicators
img: assets/img/quantops-drl.png
importance: 1
category: "QuantOps"
---

An advanced deep reinforcement learning trading system using Double Deep Q-Networks (DDQN) with multi-modal data processing for automated trading decisions. The system combines price data, technical indicators, and news sentiment analysis to execute trades across multiple assets with realistic transaction costs and risk management.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/quantops-drl.png" title="Deep RL Trading Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Accumulated returns of the deployed agents for Singapore Brent Futures.
</div>

## Architecture
- **Environment**: Custom Gymnasium-based trading environments with realistic market dynamics
- **Agent**: DDQN with experience replay and target networks for stable learning
- **Data Processing**: Multi-modal pipeline integrating price, technical indicators, and semantic analysis

## Key Features
**Double Deep Q-Network Agent**: Experience replay buffer (10,000 capacity), separate policy and target networks updated every 500 steps, Huber loss with Adam optimizer (1e-4 learning rate, 1e-5 L2 regularization), epsilon-greedy exploration (start: 0.9, min: 0.05, decay: 0.995), batch size: 64.

**Trading Environment**: Discrete action space with 11 levels (SELL/HOLD/BUY at 5 intensity levels: 10%, 20%, 40%, 60%, 80%, 100%), multi-dimensional observation space (num_assets × window_size × features), realistic transaction costs (0.2% buy, 0.3% sell matching IPOT fees), maximum 30% portfolio allocation per asset.

**Reward Engineering**: Sharpe Ratio-based rewards for risk-adjusted returns, mid-term look-ahead returns (configurable 10-step horizon), immediate next-step returns, and window-based observation system (default: 10 timesteps).

**Multi-Modal Data Integration**: News sentiment analysis using DistilRoBERTa for real-time market sentiment, technical indicator processing with automated feature extraction, and preprocessing pipeline for normalization.

**Training Variants**: Baseline DDQN training, TimesNet-enhanced DDQN for temporal pattern recognition, multi-agent DDQN for collaborative strategies, rolling window training for adaptive learning, and experimental configurations.

**Performance Monitoring**: Episode-based reward tracking, total profit calculation, asset value history visualization using Seaborn, current holdings and balance reporting, and training progress callbacks.

## Technical Stack
- PyTorch for deep learning and neural network architectures
- Gymnasium for custom RL environments
- Stable-Baselines3 for environment utilities
- Transformers (Hugging Face) for sentiment analysis
- NumPy for numerical computations
- Matplotlib/Seaborn for visualization

## Performance Metrics
- Total portfolio return tracking
- Sharpe ratio for risk-adjusted performance
- Win rate and trade count analysis
- Maximum drawdown monitoring
- Asset value trajectory visualization
