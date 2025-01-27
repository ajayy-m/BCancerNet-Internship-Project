import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Neural Network model setup (same as before)
def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Predictive system
def predict_breast_cancer(input_data, model, scaler, feature_names):
    # Convert input data to a NumPy array
    input_data_as_numpy_array = np.asarray(input_data)
    # Reshape the array for a single sample
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    # Convert to DataFrame with proper column names
    input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)
    # Standardize the input data
    input_data_std = scaler.transform(input_data_df)
    # Perform prediction
    prediction = model.predict(input_data_std)
    prediction_label = np.argmax(prediction)
    
    if prediction_label == 0:
        return "The tumor is Malignant"
    else:
        return "The tumor is Benign"

# Streamlit App with enhanced website-like theme
st.set_page_config(page_title="Breast Cancer Prediction App", page_icon="🏥", layout="centered")

# Add custom CSS for decoration (website-like theme)
st.markdown(
    """
    <style>
        body {
            background-color: #e9ecef;
            color: #343a40;
            font-family: 'Roboto', sans-serif;
        }
        
        .stButton button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .stButton button:hover {
            background-color: #0056b3;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }

        h1, h2, h3 {
            font-family: 'Merriweather', serif;
            color: #007bff;
        }

        .stDataFrame {
            margin-top: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .stSidebar .sidebar-content {
            background-color: #f8f9fa;
        }

        .stTextInput input {
            font-size: 16px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Breast Cancer Prediction App")

st.write("""
    This app predicts whether a tumor is **Malignant** or **Benign** based on the provided tumor features.
    Upload your dataset and make predictions with the help of machine learning!
""")
st.warning("""
    **Note:** Ensure that your uploaded dataset contains all the required feature names as columns. 
    The feature names must match exactly with those expected by the model for accurate predictions.
""")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV file into DataFrame
    data_frame = pd.read_csv(uploaded_file)
    
    # Display first few rows to user for confirmation
    st.write("Here is the data you uploaded:")
    st.dataframe(data_frame.head())

    # Ensure the last column is the target (label)
    if 'label' in data_frame.columns:
        X = data_frame.drop(columns='label')
        Y = data_frame['label']

        # Split dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

        # Standardize the data
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        # Build and train the model
        model = build_model(input_shape=(X_train.shape[1],))
        history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

        # Sidebar with graphs and analytics
        st.sidebar.title("Analytics and Graphs")

        # Accuracy plot
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Training Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_title('Model Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.sidebar.pyplot(fig)

        # Loss plot
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.sidebar.pyplot(fig)

        # User inputs for prediction
        st.header("Enter Tumor Features")
        user_inputs = []
        feature_names = list(X.columns)
        for feature_name in feature_names:
            value = st.number_input(f"Enter {feature_name}:", min_value=0.0, format="%.4f")
            user_inputs.append(value)

        if st.button("Predict"):
            if any(user_inputs):
                result = predict_breast_cancer(user_inputs, model, scaler, feature_names)
                st.subheader("Prediction Result:")
                st.success(result)
            else:
                st.warning("Please enter all the required features to make a prediction.")
    else:
        st.error("The CSV file must contain a 'label' column for classification.")
else:
    st.write("Please upload a CSV file to get started.")
