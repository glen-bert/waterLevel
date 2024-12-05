import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Map column names to model file names
model_map = {
    "Tanaon": "tanaon_lstm_attention_2.h5",
    "Guihean": "guihean_lstm_attention_2.h5",
    "Upper Amusig": "upper_amusig_lstm_attention_2.h5",
    "Lower Amusig": "lower_amusig_lstm_attention_2.h5"
}

# Load a model based on column selection
@st.cache_resource
def load_trained_model(model_file):
    return load_model(model_file)

# Function to scale data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

# Function to make predictions
def make_predictions(model, scaled_data):
    predicted_level = []
    current_batch = scaled_data.reshape(1, 60, 1)

    for _ in range(24):  # Predicting 24 hours
        next_prediction = model.predict(current_batch, verbose=0)
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
        predicted_level.append(next_prediction)
    return predicted_level

# Streamlit App
st.title("24-Hour Water Level Forecasting")

st.write("""
Upload data (last 60 hours of Water Level values) as a CSV file. The model will predict the next 24-hour values using the corresponding model.
""")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Select a parameter for prediction
        parameter = st.selectbox(
            "Select a parameter for prediction:",
            options=list(model_map.keys())
        )

        if parameter not in df.columns:
            st.error(f"The selected parameter '{parameter}' is not in the uploaded file.")
        else:
            # Extract the last 60 days of data for the selected parameter
            parameter_data = df[parameter].values[-60:]
            if len(parameter_data) != 60:
                st.error(f"The data must contain at least 60 rows for '{parameter}' to make predictions.")
            else:
                # Scale data
                scaled_data, scaler = scale_data(parameter_data)

                # Load the corresponding model
                model_file = model_map[parameter]
                model = load_trained_model(model_file)

                # Make predictions
                predictions = make_predictions(model, scaled_data)

                # Inverse scale predictions
                predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

                # Display results
                st.write(f"### Predicted Levels for Next 24 Hours of {parameter}")
                result_df = pd.DataFrame({
                    "Hour": list(range(1, 25)),
                    "Predicted Level": predicted_values
                })
                st.dataframe(result_df, width=800, height=400)

                st.line_chart(predicted_values)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to start.")
