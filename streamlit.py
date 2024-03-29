import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO


# Define the Streamlit app
def main():
    st.title("Traffic Detection App")

    # Upload file through Streamlit's file uploader
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Make a POST request to the FastAPI endpoint
    if file is not None:
        response = requests.post("http://localhost:8000/predict/", files={"file": file})
        if response.status_code == 200:
            # Display the uploaded image
            image = Image.open(file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Load the processed image from the response content
            processed_image = Image.open(BytesIO(response.content))
            st.image(processed_image, caption="Processed Image", use_column_width=True)  
        else:
            st.error("Error occurred during prediction. Please try again.")

    # Run the Streamlit app
if __name__ == "__main__":
    main()