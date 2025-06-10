import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from base_det import BaseDet


class LicensePlateDetection(BaseDet):
    def __init__(self):
        super().__init__()
        self.option_name = "License Plate Detection"

    def run(self):
        self.upload_image()
        # Predict button
        response = None
        if st.button("Generate result"):
            with st.spinner("Processing..."):
                # Placeholder for ML prediction
                # You will implement the actual prediction here
                payload = {"type": "license_plate"}
                files = {"item": ("image.jpg", self.buffer, "image/jpeg")}

                # attach image to files
                response = requests.post(self._PREDICT_URL, files=files, data=payload)
                # print(response.json())

        # Display output section
        st.subheader("Result")
        if response:
            output_image = Image.open(BytesIO(response.content))
            st.image(
                output_image,
                caption="Processed Image",
                use_container_width=False,
                width=self._WIDTH,
            )
            print("Get response successfully!")
