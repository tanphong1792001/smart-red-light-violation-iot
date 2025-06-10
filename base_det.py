from abc import ABC, abstractmethod
import streamlit as st
from PIL import Image
from io import BytesIO
import cv2

class BaseDet(ABC):
    def __init__(self):
        self.option_name = "Base Detection"
        self._WIDTH = 1200
        self._PREDICT_URL = "http://127.0.0.1:8000/predict_single"
        self.buffer = BytesIO()
        self.image = None

    def upload_image(self):
        st.title(f"{self.option_name}")

        # File uploader for images
        uploaded_file = st.file_uploader(
            f"Upload an image for {self.option_name}", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            # Display the uploaded image
            self.image = Image.open(uploaded_file)
            # Save uploaded file to a temporary file
            if self.image.format == "PNG":
                self.image = self.image.convert("RGB")
            self.image.save(self.buffer, format="JPEG")
            self.buffer.seek(0)
            st.image(
                self.image,
                caption="Uploaded Image",
                use_container_width=False,
                width=self._WIDTH,
            )

    def upload_video(self):
        st.title(f"{self.option_name}")

        # File uploader for videos
        self.uploaded_file = st.file_uploader(
            f"Upload a video for {self.option_name}", type=["mp4", "avi"]
        )

        if self.uploaded_file is not None:
            # Display the uploaded video
            st_video = st.video(self.uploaded_file)
    

    @abstractmethod
    def run(self):
        pass
