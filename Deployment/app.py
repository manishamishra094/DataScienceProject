import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Country Clustering using KMeans")

st.info(
    "Upload a CSV file with the same 23 numerical features used during model training."
)

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV
        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(input_df.head())

        # Converting DataFrame to NumPy array before scaling
        scaled_input = scaler.transform(input_df.values)

        # Predict clusters
        predictions = model.predict(scaled_input)

        # Add cluster labels
        input_df['Cluster'] = predictions

        st.success("Clustering completed successfully!")
        st.subheader("Clustered Data")
        st.dataframe(input_df)

    except Exception as e:
        st.error(" Error while processing the file.")
        st.write(e)
