import streamlit as st
import pandas as pd
import gzip
import pickle

# Function to load a gzip-compressed model
def load_model(filepath):
    with gzip.open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

# Load the model
model = load_model('model_binary.dat.gz')

# Streamlit app
st.title("Disease Prediction App")

st.write("""
# Disease Prediction
This app predicts the probability of a disease based on user inputs!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    feature1 = st.sidebar.slider('Feature 1', min_value=0, max_value=100, value=50)
    feature2 = st.sidebar.slider('Feature 2', min_value=0.0, max_value=10.0, value=5.0)
    feature3 = st.sidebar.selectbox('Feature 3', options=[0, 1])
    data = {'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

try:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    st.write('Disease' if prediction[0] else 'No Disease')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
except AttributeError as e:
    st.error(f'Error: {e}')
except Exception as e:
    st.error(f'Unexpected error: {e}')

# Ensure that your model is indeed the correct object
st.write(type(model))
