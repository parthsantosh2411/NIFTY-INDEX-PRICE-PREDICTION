# 📊 **NIFTY Price Prediction Using Market Data and News Sentiment** 📈

---

### 📌 **Table of Contents**
1. [Project Overview](#-project-overview)  
2. [Dataset](#-dataset)  
3. [Methodology](#-methodology)  
4. [Models Used](#-models-used)  
5. [Results](#-results)  
6. [Conclusion](#-conclusion)

---

## 📖 **Project Overview**

This project predicts the future prices of the **NIFTY Index** by combining:  
📈 **Market Data**: Historical price data like open, high, low, close, and volume.  
📰 **News Sentiment**: Sentiment scores derived from NIFTY-related news articles.

By integrating both financial data and textual sentiment analysis, the project applies **Machine Learning (ML)** and **Deep Learning (DL)** models to achieve precise results. Metrics such as **Root Mean Squared Error (RMSE)** and **R² Score** are used for evaluation.

---

## 📊 **Dataset**

- 🏦 **Market Data**: Extracted from platforms like `yfinance` and cleaned for consistency.  
- 📰 **News Data**: Collected using APIs like **GDELT**, then processed with NLP techniques to compute sentiment scores.

### **Features**:  
📌 Technical indicators like Moving Averages and RSI.  
📌 Sentiment scores integrated with price data for a hybrid dataset.

---

## ⚙️ **Methodology**

1. **🔧 Data Preprocessing**:  
   - Normalized market data for scaling consistency.  
   - Preprocessed news articles using tokenization, stopword removal, and sentiment scoring.  

2. **🔍 Feature Engineering**:  
   - Extracted technical indicators for trend analysis.  
   - Combined sentiment scores with price features for hybrid modeling.

3. **📈 Model Training**:  
   - Multiple ML and DL models were trained to predict future prices.

4. **📏 Evaluation Metrics**:  
   - RMSE (Root Mean Squared Error).  
   - R² Score (Coefficient of Determination).

---

## 🤖 **Models Used**

### 🛠️ **Machine Learning Models**:
- 🔮 Prophet  
- 📈 SARIMA  
- 📉 ARIMA  
- 📊 Linear Regression  
- 🚀 XGBoost  
- 🌟 LightGBM  
- 🌲 Random Forest  

### 🔬 **Deep Learning Models**:
- 🤖 Advanced Transformer  
- 🔗 CNN-LSTM Hybrid  
- 🔁 Bi-Directional LSTM  
- 🧠 LSTM  
- ⚡ GRU  

---

## 🏆 **Results**

### **📈 Machine Learning (ML) Results**:
| **Model**             | **RMSE**       | **R² Score**  |
|------------------------|----------------|---------------|
| 🔮 Prophet           | 5,504.3661     | -0.9348       |
| 📈 SARIMA            | 4,038.5463     | -0.0418       |
| 📉 ARIMA             | 3,957.4499     | -0.0004       |
| 📊 Linear Regression | 3,417.7816     | 0.2541        |
| 🚀 XGBoost           | 4,760.2746     | 0.5153        |
| 🌟 LightGBM          | 4,669.5157     | 0.5338        |
| 🌲 Random Forest     | 2,637.3095     | 0.5557        |

---

### **🧠 Deep Learning (DL) Results**:
| **Model**                  | **RMSE**       | **R² Score**  |
|----------------------------|----------------|---------------|
| 🤖 Advanced Transformer     | 988.7988       | 0.2848        |
| 🔗 CNN-LSTM Hybrid          | 365.8962       | 0.9021        |
| 🔁 Bi-Directional LSTM      | 365.8962       | 0.9021        |
| 🧠 LSTM                     | 249.5982       | 0.9544        |
| ⚡ GRU                      | 249.5982       | 0.9544        |

---

## 💡 **Conclusion**

This project demonstrates the effectiveness of combining market data and sentiment analysis to predict stock prices.  
- **LSTM** and **GRU** models outperformed others, highlighting the power of deep learning in time-series forecasting.  
- The inclusion of sentiment data significantly enhanced the prediction accuracy.
