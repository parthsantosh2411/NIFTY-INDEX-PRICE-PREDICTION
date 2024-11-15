import streamlit as st
from PIL import Image
import pandas as pd
from datetime import timedelta

# Page setup with aesthetics
st.set_page_config(page_title="📈 NIFTY Price Prediction App", page_icon="📉", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #1a1a1d;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
    }
    .stSidebar {
        background-color: #e1e4eb;
    }
    .stButton button {
        background-color: #1a73e8;
        color: white;
        font-size: 1.1em;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ("Home", "Performance Visualization", "Model Metrics", "Future Forecast"))

# Home Section
if section == "Home":
    st.title("📉 NIFTY Price Prediction App")
    st.write("""
    Welcome to the NIFTY Price Prediction App—an innovative platform that leverages sentiment analysis and advanced modeling techniques to forecast NIFTY 50 index prices.

## Project Overview
             
This app is based on a comprehensive study that combines news sentiment analysis with machine learning and deep learning models to predict the NIFTY 50 index closing price. By analyzing five years of historical data and sentiment scores derived from news articles, this project explores how financial markets can be influenced by public sentiment and how models can leverage this information for more accurate predictions.

## Why Sentiment Analysis?
             
Market sentiment often reflects the collective emotional response to economic and geopolitical events, which directly impacts stock market performance. Incorporating news sentiment scores enables our models to better capture this intangible aspect, enhancing prediction accuracy for the NIFTY 50 index.

## Models Utilized
             
Our approach integrates traditional machine learning models and advanced deep learning architectures to evaluate the best-performing methods for NIFTY 50 forecasting:

## Traditional Machine Learning Models:

Linear Regression
Random Forest
ARIMA & SARIMA
Prophet
XGBoost
LightGBM
Deep Learning Models:

LSTM (Long Short-Term Memory)
Bidirectional LSTM
CNN-LSTM (Convolutional LSTM)
GRU (Gated Recurrent Unit)
Transformer
             
Key Findings
Our results demonstrate that deep learning models, particularly GRU and LSTM, outperform traditional machine learning models by a significant margin. By effectively capturing temporal dependencies and sentiment dynamics, these models achieved higher accuracy, underscoring the benefits of sentiment-enhanced deep learning for stock market prediction.

## How to Use the App
             
Navigate through the sidebar to explore:

Performance Visualization: Compare actual vs. predicted NIFTY prices across various models.
             
Model Metrics: Review R-squared, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) scores for each model.
             
Future Forecast: Make your own predictions for the NIFTY index based on your chosen model and forecast period.
             
Dive in and see how advanced analytics and sentiment insights can bring precision to NIFTY index forecasting. Empower your market predictions with the power of data and sentiment analysis!
    """)

# Performance Visualization Section
elif section == "Performance Visualization":
    st.title("📊 Performance Visualization")
    st.write("### Actual vs. Predicted Stock Prices")

    # Define paths for saved images, with ML models first ordered by R² Score from lowest to highest, then DL models
    image_paths = {
        "Prophet": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\Prophet.png",
        "SARIMA": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\Sarima.png",
        "ARIMA": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\Arima.png",
        "Linear Regression": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\output.png",
        "XGBoost": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\XGboost.png",
        "LightGBM": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\LightGBM.png",
        "Random Forest": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\Random Forest.png",
        "Advanced Transformer": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\Advance Transformer.png",
        "CNN-LSTM Model": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\Cnn-lstm Model.png",
        "Bi-directional LSTM": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\bi-directional lstm.png",
        "LSTM": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\Lstm.png",
        "GRU": "C:\\Users\\Anshul\\OneDrive\\Videos\\Images\\GRU.png"
    }

    # Model selection dropdown
    model_choice = st.selectbox("Select Model for Visualization", list(image_paths.keys()))

    # Display selected image
    image = Image.open(image_paths[model_choice])
    st.image(image, caption=f"{model_choice} Model: Actual vs Predicted Close Prices", use_container_width=True)

# Model Metrics Section
elif section == "Model Metrics":
    st.title("📐 Model Metrics")
    st.write("### Root Mean Squared Error & R² Score")

    # Model metrics data, with ML models first in ascending R² Score, then DL models
    metrics_data = {
        "Model": [
            "Prophet", "SARIMA", "ARIMA", "Linear Regression", 
            "XGBoost", "LightGBM", "Random Forest", 
            "Advanced Transformer", "CNN-LSTM Hybrid", "Bi-directional LSTM", "LSTM", "GRU"
        ],
        "Root Mean Squared Error": [
            5504.3661, 4038.5463, 3957.4499, 3417.7816, 4760.2746, 
            4669.5157, 2637.3095, 988.7988, 365.8962, 365.8962, 249.5982, 249.5982
        ],
        "R² Score": [
            -0.9348, -0.0418, -0.0004, 0.2541, 0.5153, 
            0.5338, 0.5557, 0.2848, 0.9021, 0.9021, 0.9544, 0.9544
        ]
    }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Display the table
    st.table(metrics_df)

# Future Forecast Section
elif section == "Future Forecast":
    st.title("🔮 Future Forecast")
    st.write("### Predict Future NIFTY Index Prices")

    # User inputs
    model_choice = st.selectbox("Select a model for prediction", (
        "Linear Regression", "Random Forest", "ARIMA", "SARIMA", "XGBoost", 
        "LightGBM", "Advanced Transformer", "LSTM", "GRU", "CNN-LSTM", 
        "Bi-directional LSTM"
    ))
    days_to_predict = st.number_input("Enter the number of days to predict:", min_value=1, max_value=30, step=1)

    if st.button("Predict"):
        # Generate future dates without the time component
        start_date = pd.to_datetime('2024-11-01')
        future_dates = [start_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
        future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]  # Format dates without time
        
        # Hardcoded predicted values for the next 30 days for each model
        model_predictions = {
            "Linear Regression": [
                16900.871852, 16877.203778, 16886.225899, 16879.457567, 16886.049799,
                16878.991760, 16886.156904, 16897.911529, 16891.847902, 16876.397393,
                16893.984273, 16911.183439, 16881.906097, 16908.647315, 16899.756900,
                16887.568511, 16880.191040, 16910.350810, 16886.371379, 16881.389066,
                16904.119276, 16890.434517, 16889.477792, 16885.313688, 16899.415743,
                16864.326003, 16902.289881, 16918.805108, 16884.587206, 16890.767749
            ],
            "Random Forest": [
                21996.470223, 21999.227961, 22002.560128, 21997.058326, 22006.900306,
                21992.262539, 22011.741319, 21994.875316, 22002.741342, 21991.526740,
                21994.896115, 22004.945258, 22007.888490, 22002.272612, 22000.580577,
                21998.068969, 21989.796652, 22003.855492, 22002.233555, 22014.033069,
                22011.134544, 21979.699761, 22016.487599, 21998.976859, 22008.348902,
                22004.055078, 21988.986507, 22003.509669, 21999.007374, 22020.305923
            ],
            "ARIMA": [
                14981.912663, 14978.040500, 14979.202563, 14978.037057, 14987.787067,
                14981.568276, 14988.906753, 14963.126469, 14985.110880, 14977.962035,
                14975.098818, 14989.407387, 14977.147444, 14980.088141, 14978.220251,
                14974.711741, 14970.680965, 14982.561947, 14986.271588, 14975.958589,
                14988.187495, 14966.798407, 14977.686359, 14984.545351, 14967.813535,
                14986.406865, 14971.822198, 14974.786381, 14996.474415, 14971.918290
            ],
            "SARIMA": [
                21455.533785, 21447.015562, 21478.132444, 21442.131372, 21471.899512,
                21447.794399, 21453.215518, 21455.816592, 21441.325285, 21450.299416,
                21462.545189, 21447.794199, 21443.536373, 21447.755554, 21449.191244,
                21435.010626, 21448.568920, 21456.355915, 21441.236422, 21447.713737,
                21433.525141, 21471.021530, 21451.212674, 21463.212537, 21451.151473,
                21430.979662, 21431.777493, 21438.844158, 21449.786320, 21442.522217
            ],
            "XGBoost": [
                24315.547035, 24302.876720, 24205.693823, 24299.317060, 24303.498962,
                24344.558513, 24270.937222, 24295.935894, 24313.109310, 24320.677535,
                24297.499359, 24322.388549, 24251.205400, 24345.999942, 24298.119000,
                24299.556017, 24279.715690, 24259.280754, 24323.391474, 24299.215060,
                24330.755029, 24366.880807, 24270.403241, 24344.517554, 24307.869965,
                24317.422455, 24283.952965, 24336.476315, 24273.266097, 24256.467241
            ],
            "LightGBM": [
                17207.572496, 17209.014464, 17196.657250, 17223.842419, 17197.680005,
                17197.000090, 17186.033709, 17190.384733, 17209.024523, 17203.114315,
                17185.099510, 17217.318516, 17195.744038, 17206.396460, 17189.371310,
                17205.630135, 17199.774718, 17189.796506, 17201.386300, 17194.460549,
                17201.593957, 17211.915301, 17202.064383, 17198.505555, 17164.374061,
                17190.487050, 17184.112115, 17182.689334, 17196.920735, 17187.933215
            ],
            "LSTM": [
                24143.829863, 24113.983048, 24144.378742, 24185.406743, 24184.687032,
                24163.612709, 24132.925135, 24142.719928, 24175.136821, 24166.728140,
                24159.496864, 24096.286656, 24184.374818, 24180.203111, 24159.882780,
                24144.354327, 24158.324356, 24184.977877, 24173.174459, 24156.976028,
                24160.897009, 24130.711860, 24145.171222, 24194.494837, 24149.626480,
                24174.913226, 24156.749631, 24130.636035, 24148.120919, 24099.344103
            ],
             "GRU": [
                24039.144033, 24060.259924, 24038.937835, 24014.013930, 24058.863750,
                23997.784299, 24092.846443, 24045.860872, 24056.236392, 24089.231719,
                24039.899550, 24121.095544, 24027.469660, 24129.587361, 24072.653872,
                24036.423617, 23986.111720, 24052.514336, 24042.956699, 24147.544466,
                24010.514621, 24040.224364, 24115.517341, 24071.819145, 24069.464790,
                24018.849985, 23950.566947, 24032.540070, 24078.154255, 24123.735428
            ],
            "CNN-LSTM": [
                24315.547035, 24302.876720, 24205.693823, 24299.317060, 24303.498962,
                24344.558513, 24270.937222, 24295.935894, 24313.109310, 24320.677535,
                24297.499359, 24322.388549, 24251.205400, 24345.999942, 24298.119000,
                24299.556017, 24279.715690, 24259.280754, 24323.391474, 24299.215060,
                24330.755029, 24366.880807, 24270.403241, 24344.517554, 24307.869965,
                24317.422455, 24283.952965, 24336.476315, 24273.266097, 24256.467241
            ],
            "Bi-directional LSTM": [
                24988.428214, 25016.392536, 24978.919119, 25103.212563, 24975.664365,
                24951.674195, 25039.682663, 25010.951242, 25002.467395, 25047.338063,
                25009.240124, 25033.890109, 25003.096553, 25033.061740, 25017.596163,
                25011.631457, 24996.944327, 25026.158984, 25030.152907, 25009.568014,
                24984.415701, 25059.783461, 25038.757305, 25032.462948, 25042.223815,
                25006.699191, 25051.053436, 24969.076364, 25023.596356, 24963.282518
            ]

            # Add other model hardcoded predictions similarly
        }
        
        # Select the predictions based on the model choice
        predicted_values = model_predictions[model_choice][:days_to_predict]
        
        # Create a DataFrame with these dates and values
        results = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close": predicted_values
        })
        results.set_index("Date", inplace=True)

        # Display predictions
        st.write(f"### {days_to_predict}-Day Predicted Prices Using {model_choice} Model")
        st.dataframe(results)
        st.line_chart(results["Predicted Close"])
