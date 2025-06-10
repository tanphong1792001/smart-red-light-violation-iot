import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from base_det import BaseDet
import tempfile
import os

class RedLightViolation(BaseDet):
    def __init__(self):
        super().__init__()
        self.option_name = "Red Light Violation Detection"
        self._PREDICT_URL = "http://127.0.0.1:8000/red_light_violation"

    def run(self):
        self.upload_video()
        response = None
        if st.button("Generate result"):
            with st.spinner("Processing.... May take a few minutes!"):
                video_bytes = self.uploaded_file.read()

                headers = {"Content-Type": "application/octet-stream"}
                response = requests.post(self._PREDICT_URL, data=video_bytes, headers=headers)
                    

        # Display output section
        if not response:
            st.warning("Please upload a video and click 'Generate result'.")
            return

        if response.status_code == 200:
            # Save response content (video bytes) to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(response.content)
                temp_video_path = tmp_file.name
            st.subheader("🎬 Processed Video")
            st.video(temp_video_path)

            # delete temp video
            os.remove(temp_video_path)


        else:
            st.error(f"❌ Processing failed: {response.status_code}")
            st.text(response.text)
