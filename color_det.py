import cv2
import numpy as np
import streamlit as st

from base_det import BaseDet


class ColorDetection(BaseDet):
    def __init__(self):
        super().__init__()
        self.option_name = "Color Detection"

    def run(self):
        image_hsv = None
        self.upload_image()
        if self.image is not None:
            image_hsv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2HSV)

        col1, col2 = st.columns(2)
        with col1:
            # three sliders for hue
            st.title("Hue Lower Range")
            hue_lower_first = st.slider("LR1", 0, 255, 0)
            hue_lower_second_ = st.slider("LR2", 0, 255, 0)
            hue_lower_third = st.slider("LR3", 0, 255, 0)

        with col2:
            # three sliders for hue
            st.title("Hue Upper Range")
            hue_upper_first = st.slider("UR1", 0, 255, 255)
            hue_upper_second_ = st.slider("UR2", 0, 255, 255)
            hue_upper_third = st.slider("UR3", 0, 255, 255)

        st.text(
            "Example for Hue Range of green for test_crossroad.png: Lower [0, 155, 220], Upper [255, 255, 255]"
        )
        lower_range = np.array([hue_lower_first, hue_lower_second_, hue_lower_third])
        upper_range = np.array([hue_upper_first, hue_upper_second_, hue_upper_third])

        if image_hsv is not None:
            # Create a mask for the selected color range
            mask = cv2.inRange(image_hsv, lower_range, upper_range)
            # Apply the mask to the original image
            result = cv2.bitwise_and(
                np.array(self.image), np.array(self.image), mask=mask
            )

            # Convert the result back to RGB
            result_rgb = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
            # Display the result

            st.image(
                result_rgb,
                caption="Processed Image",
                use_container_width=False,
                width=self._WIDTH,
            )
