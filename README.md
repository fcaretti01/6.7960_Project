# Decoding Multi-Head Attention for Intra-Day Financial Time-Series Forecasting

## Introduction
This project investigates the application of multi-head self-attention mechanisms, specifically in the context of GPT-2 architecture, for forecasting high-frequency financial time series data. The objective is to interpret how transformer-based models, particularly those using multi-head attention, prioritize various features and dependencies in the input sequence when predicting future market behavior.

## Project Overview
The project focuses on decoding the decision-making process of GPT-2, specifically its use of multi-head attention, for predicting the sign of the return in intraday financial markets. The model uses a set of lagged input features to perform autoregressive predictions in a three-class classification problem. We analyze the internal workings of GPT-2, applying circuit analysis techniques and "path patching" methodology to trace the flow of information within the model.

Key highlights of this project include:
- **Multi-Head Attention:** Understanding how different attention heads specialize in capturing long-term versus short-term patterns in time series data.
- **GPT-2 Architecture:** Employing a transformer-based model, GPT-2, and adapting it for financial time-series forecasting.
- **Model Interpretability:** Using mean ablation and attention activation analysis to interpret the importance of individual attention heads.

## Authors
- Filippo Caretti (fcaretti@mit.edu)
- Ludovico Ghitturi (ludo25@mit.edu)

## Project Outline
1. **Introduction**: Overview of the project's focus and objectives in financial time-series forecasting using transformer models.
2. **Background**: Discusses the self-attention mechanism in transformers and how it enables the model to capture both local and global dependencies in sequence data.
3. **GPT-2 Architecture**: A detailed explanation of GPT-2’s design, including the masked self-attention mechanism and its autoregressive prediction capabilities.
4. **Task Definition**: Describes the task of predicting the sign of the return for 30-minute intervals using lagged input features.
5. **Network Analysis**: Investigation of the model's inner workings, focusing on the contribution of each attention head using ablation studies.
6. **Implementation**: Overview of the steps taken in data collection, preparation, model training, and network analysis.
7. **Results**: Presents the findings from training two models with different dimensionalities and the impact of attention heads on model predictions.
8. **Conclusion**: Summarizes the insights gained from the project, including the relevance of attention head specialization and model dimensionality.

## Technologies Used
- **GPT-2 Architecture** (Adapted for time-series forecasting)
- **Python** (for model implementation and training)
- **PyTorch** (for deep learning model training)
- **Matplotlib** and **Seaborn** (for visualizations)
- **Polygon.io API** (for financial data collection)





## Results
- **Attention Heads**: We identified two key classes of attention heads: the **Time Persistence Head** and the **Intra-Day Dynamics Head**, which focus on long-term and short-term patterns in the data, respectively.
- **Performance**: Both small and large model variants were tested, with larger models showing improved predictive capabilities, especially in terms of time persistence patterns.

## References
1. [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593) - Kevin Wang et al.
2. [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) - Kevin Meng et al.
3. [Modern Methods of Text Generation](https://arxiv.org/pdf/2009.04968) - Montesinos, D. M. (2020)
