import streamlit as st
import pandas as pd
import joblib
import numpy as np

import pickle

# Load SVM model
model = joblib.load('Classification.pkl')

with open('scaler.pkl', 'rb') as file:
    sc = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Function to predict using the SVM model
def predict(features):

    features = np.array(features).reshape(1, -1)
    c_std = sc.transform(features)
    # Predict the class using the pre-trained SVM model
    predicted_class = model.predict(c_std)

    return predicted_class



# Streamlit UI
def main():
    st.title('BP ANALYSIS')

    # Input features
    feature1 = st.number_input('BP')
    feature2 = st.number_input('Age')    

    # Predict button
    if st.button('Predict'):
        features =[feature1, feature2]

        prediction = predict(features)

        predicted_class_decoded = label_encoder.inverse_transform(prediction)

        # Join the predicted class values into a string
        arr_str_flat = ', '.join(predicted_class_decoded.flatten())

        # Print the recommended medicine based on the predicted class
        st.write("Recommended Medicine:", arr_str_flat)

        # Determine the recommended dosage level based on the predicted class
        if prediction == 0:
            if feature2 <= 18:
                st.write("Recommended Dosage Level: 5mg")
            else:
                st.write("Recommended Dosage Level: 10mg")
        elif prediction == 1:
                st.write("Recommended Dosage Level: 2.5mg")
        else:
             st.write("The Patient is Normal")


if __name__ == '__main__':
    main()
