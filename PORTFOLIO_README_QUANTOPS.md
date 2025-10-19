# Intelligent Trading Decision Support System

A comprehensive suite of AI-powered financial analysis applications designed to support trading decisions through advanced machine learning and deep reinforcement learning techniques. This portfolio demonstrates end-to-end implementation of three distinct products for financial market analysis.

---

## Product Overview

### 1. Trend Prediction

A full-stack application that predicts market sentiment and categorizes financial news using natural language processing and machine learning.

#### Architecture
- **Backend**: Spring Boot (Java) with Gradle build system
- **Frontend**: Next.js with TypeScript and Tailwind CSS
- **ML Model**: Python-based prediction service with dual classification system

#### Key Features

**Semantic Analysis Engine**
- Utilizes DistilRoBERTa transformer model for sentiment prediction
- Classifies news headlines into three sentiment categories: Bearish, Neutral, Bullish
- Outputs confidence scores for each sentiment using sigmoid normalization
- GPU-accelerated inference with CUDA support

**News Categorization**
- Automated topic classification for financial news headlines
- Multi-class classification system for organizing news by subject matter
- Integrated topic verification workflow

**Technical Implementation**
- Model initialization and loading via centralized model management system
- Batch prediction pipeline supporting DataFrame operations
- Token encoding with RobertaTokenizerFast with 512 max token length
- Attention mask generation for improved model accuracy

**MLOps Pipeline**
- Experiment tracking with MLflow integration
- Automated cross-validation for model evaluation
- Hyperparameter optimization using Optuna TPE sampler
- Time series forecasting models: TSMixer, TSMixerx, NBEATSx, NHITS
- Model versioning and checkpoint management
- Artifact logging for reproducibility

#### Technical Stack
- Transformers (Hugging Face)
- PyTorch with CUDA support
- Pandas for data manipulation
- NeuralForecast for time series modeling
- MLflow for experiment tracking

---

### 2. Price Forecasting

A machine learning application for forecasting commodity and financial asset prices using state-of-the-art time series neural networks with automated hyperparameter tuning.

#### Architecture
- Modular ML pipeline with separate ingestion, transformation, and prediction components
- Automated experiment tracking and model versioning
- Support for both univariate and multivariate time series forecasting

#### Key Features

**Forecasting Models**
- **AutoNHITS**: Neural Hierarchical Interpolation for Time Series with automated architecture search
- **AutoTSMixerx**: Time Series Mixer with exogenous variables support
- **AutoiTransformer**: Inverted Transformer architecture for multivariate forecasting
- **AutoTSMixer**: Standard Time Series Mixer for multivariate analysis

**Hyperparameter Optimization**
- Automated hyperparameter search using Optuna TPE sampler
- Configurable search spaces for input window sizes and model parameters
- Multiple trial sampling (configurable, default 10 samples per model)
- Early stopping based on validation loss (patience: 5 steps)

**Data Processing Pipeline**
- Calendar feature engineering: day_of_week, is_weekend, month, day_of_month, quarter, year, is_holiday
- Standard scaling for univariate forecasting
- Robust scaling for multivariate forecasting
- Automatic train/validation/test split (90%/5%/5% for univariate, 80%/10%/10% for multivariate)

**Technical Analysis Integration**
- Moving averages (customizable window lengths)
- Relative Strength Index (RSI) with configurable periods
- Bollinger Bands (customizable length and standard deviation)
- MACD (Moving Average Convergence Divergence) with signal line
- Ultimate Oscillator (multi-timeframe momentum indicator)
- ADX (Average Directional Index) with directional movement indicators
- pandas_ta library integration for technical indicators

**Training Infrastructure**
- PyTorch Lightning for distributed training
- CSV logging for training metrics
- Custom logging directory configuration
- Maximum 1000 training steps per model
- Validation checks every 50 steps
- MAE (Mean Absolute Error) loss function

**Model Management**
- Unique run ID generation using UUID
- Artifact storage with organized directory structure
- Model checkpoint saving with dataset preservation
- Training results export to CSV
- Transformation pipeline serialization

#### Use Cases
- Crude oil price forecasting (Brent futures)
- Multi-asset price prediction
- Exogenous variable integration (Gas prices, DXY index, Brent futures/daily)

#### Technical Stack
- NeuralForecast for time series models
- Optuna for hyperparameter optimization
- PyTorch Lightning for training orchestration
- pandas_ta for technical analysis
- scikit-learn for preprocessing

---

### 3. AI Trading App

An advanced deep reinforcement learning trading system using Double Deep Q-Networks (DDQN) with multi-modal data processing for automated trading decisions across multiple assets.

#### Architecture
- **Environment**: Custom Gymnasium-based trading environments
- **Agents**: DDQN implementation with experience replay and target networks
- **Data Processing**: Multi-modal pipeline combining price data, technical indicators, and semantic analysis

#### Key Features

**Reinforcement Learning Framework**

**Double Deep Q-Network (DDQN) Agent**
- Experience replay buffer with prioritized sampling (10,000 capacity)
- Separate policy and target networks for stable learning
- Huber loss for robust Q-value estimation
- Adam optimizer with L2 regularization (1e-5 weight decay)
- Learning rate: 1e-4
- Epsilon-greedy exploration (start: 0.9, min: 0.05, decay: 0.995)
- Target network update frequency: 500 steps
- Batch size: 64

**Trading Environment (Gymnasium-based)**
- **Action Space**: Discrete(11) - SELL/HOLD/BUY with 5 intensity levels (10%, 20%, 40%, 60%, 80%, 100%)
  - SELL: 5 levels from -5 (sell 100%) to -1 (sell 20%)
  - HOLD: 0 (no action)
  - BUY: 5 levels from 1 (buy 20%) to 5 (buy 100%)
- **Observation Space**: Multi-dimensional tensor (num_assets × window_size × features)
- **Reward Functions**:
  - Sharpe Ratio-based reward (risk-adjusted returns)
  - Mid-term look-ahead returns (configurable horizon)
  - Immediate next-step returns
- **Transaction Costs**: Buy fee 0.2%, Sell fee 0.3% (IPOT realistic fees)
- **Risk Management**: Maximum 30% portfolio allocation per asset

**Advanced Environment Features**
- Window-based observation system (default: 10 timesteps)
- Look-ahead period for reward calculation (10 steps)
- Real-time asset memory tracking
- Portfolio value monitoring
- Trade counting and performance metrics
- Separate training and testing modes

**Multi-Modal Data Integration**

**Semantic Analysis Module**
- News headline sentiment analysis using DistilRoBERTa
- Real-time market sentiment scoring
- Integration of textual data into trading decisions

**Technical Indicator Processing**
- Automated feature extraction from price data
- Preprocessing pipeline for normalization
- Support for multiple technical indicators

**Training Variants**
- Baseline DDQN training
- TimesNet-enhanced DDQN for temporal pattern recognition
- Multi-agent DDQN for collaborative trading strategies
- Rolling window training for adaptive learning
- Experimental configurations for research

**Model Persistence**
- Model checkpoint saving with metadata
- Separate storage for model weights and training configuration
- Load/save functionality for inference deployment

**Performance Monitoring**
- Episode-based reward tracking
- Total profit calculation
- Asset value history visualization using Seaborn
- Current holdings and balance reporting
- Training progress callbacks

**Multi-Asset Support**
- Single-asset environments (focused trading)
- Multi-asset environments (portfolio optimization)
- Multi-agent single-asset (ensemble strategies)

#### Technical Stack
- PyTorch for deep learning
- Gymnasium for RL environments
- Stable-Baselines3 for environment utilities
- Transformers (Hugging Face) for NLP
- NumPy for numerical computations
- Matplotlib/Seaborn for visualization

---

## Repository Structure

```
intelligent-trading-dss/
├── trend_prediction_app/          # News sentiment and categorization
│   ├── trend-prediction-backend/   # Spring Boot API
│   ├── trend-prediction-frontend/  # Next.js UI
│   ├── trend-prediction-model/     # ML inference service
│   └── experimentation-mlops/      # Training and experimentation
│
├── price-forecasting-app/          # Time series price forecasting
│   ├── price-forecasting-ml/       # Forecasting models and pipeline
│   └── experimentation-mlops/      # Model training and optimization
│
└── ai_trading_app/                 # DRL trading system
    ├── drl_trading/                # Main RL trading implementation
    │   ├── agents/                 # DDQN agent implementations
    │   ├── env/                    # Trading environments
    │   ├── models/                 # Neural network architectures
    │   └── modules/                # Data processing modules
    └── ITS_v1/                     # Integrated Trading System v1
```

---

## Key Highlights

### Machine Learning Innovations
- **Transformer-based NLP**: Leveraging DistilRoBERTa for financial sentiment analysis
- **Neural Time Series Forecasting**: Utilizing cutting-edge models (TSMixer, iTransformer, NHITS)
- **Deep Reinforcement Learning**: Custom DDQN implementation with prioritized experience replay

### Software Engineering
- **Full-Stack Development**: Integration of Spring Boot, Next.js, and Python ML services
- **MLOps Best Practices**: Experiment tracking, model versioning, artifact management
- **Modular Architecture**: Separation of concerns across data processing, training, and inference

### Financial Domain Expertise
- **Realistic Trading Simulation**: Transaction costs, position sizing, risk management
- **Multi-Modal Analysis**: Combining price action, technical indicators, and news sentiment
- **Multiple Reward Strategies**: Sharpe ratio, look-ahead returns, immediate returns

### Research and Experimentation
- **Hyperparameter Optimization**: Automated search using Optuna
- **Multiple Environment Configurations**: Single-asset, multi-asset, multi-agent setups
- **Ablation Studies**: Baseline comparisons and experimental variants

---

## Technologies Used

**Machine Learning & AI**
- PyTorch
- Transformers (Hugging Face)
- NeuralForecast
- Stable-Baselines3
- Optuna
- scikit-learn

**Data Processing**
- pandas
- NumPy
- pandas_ta

**Experiment Tracking**
- MLflow
- PyTorch Lightning

**Backend**
- Spring Boot (Java)
- Gradle

**Frontend**
- Next.js
- TypeScript
- Tailwind CSS

**Visualization**
- Matplotlib
- Seaborn

---

## Performance Metrics

The applications track various performance indicators:

**Trend Prediction**
- Sentiment classification accuracy
- Confidence scores per prediction
- Topic categorization precision

**Price Forecasting**
- Mean Absolute Error (MAE)
- Model comparison across architectures
- Validation loss convergence

**AI Trading App**
- Total portfolio return
- Sharpe ratio
- Win rate and trade count
- Maximum drawdown
- Asset value trajectory

---

## Author

**Ryan**
Bioma AI

This portfolio demonstrates expertise in machine learning engineering, quantitative finance, and full-stack development, with a focus on practical applications of AI in financial markets.
