import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
@st.cache
def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

model = load_model()

# Define a function to make predictions
def predict_diabetes(features):
    prediction = model.predict(features)
    return prediction

# Main function to run the Streamlit app
def main():
    st.title('Diabetes Prediction App')

    # Input feature values
    Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, step=1)
    Glucose = st.number_input('Glucose', min_value=0, max_value=200, step=1)
    BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=122, step=1)
    SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=99, step=1)
    Insulin = st.number_input('Insulin', min_value=0, max_value=846, step=1)
    BMI = st.number_input('BMI', min_value=0.0, max_value=67.1, step=0.1)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, step=0.001)
    Age = st.number_input('Age', min_value=21, max_value=81, step=1)

    # Collect input features into a dataframe
    features = pd.DataFrame({'Pregnancies': [Pregnancies],
                             'Glucose': [Glucose],
                             'BloodPressure': [BloodPressure],
                             'SkinThickness': [SkinThickness],
                             'Insulin': [Insulin],
                             'BMI': [BMI],
                             'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
                             'Age': [Age]})

    # Make prediction
    if st.button('Predict'):
        prediction = predict_diabetes(features)
        if prediction[0] == 1:
            st.success('Patient is likely to have diabetes')
        else:
            st.success('Patient is not likely to have diabetes')

if __name__ == '__main__':
    main()
