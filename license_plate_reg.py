import requests
import streamlit as st
from base_det import BaseDet


class LicensePlateRecognition(BaseDet):
    def __init__(self):
        super().__init__()
        self.option_name = "License Plate Detection"
        self._PREDICT_URL = "http://127.0.0.1:8000/ocr_run"

    def run(self):
        self.upload_image()
        # Predict button
        response = None
        if st.button("Generate result"):
            with st.spinner("Processing..."):
                # Placeholder for ML prediction
                # You will implement the actual prediction here
                files = {"item": ("image.jpg", self.buffer, "image/jpeg")}

                # attach image to files
                response = requests.post(self._PREDICT_URL, files=files)

        # Display output section
        st.subheader("Result")
        if response:
            results = response.json()["result"]
            print(results)
            displayed_text = " ".join(results)
            st.write(f"## {displayed_text}")