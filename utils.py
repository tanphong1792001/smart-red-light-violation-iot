import tempfile
import streamlit as st
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random


# Function to display image upload and prediction section
def display_image_section(option_name):
    st.title(f"{option_name}")

    # File uploader for images
    uploaded_file = st.file_uploader(
        f"Upload an image for {option_name}", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Generate result"):
        with st.spinner("Processing..."):
            # Placeholder for ML prediction
            # You will implement the actual prediction here
            st.success(f"Prediction for {option_name} completed!")

    # Display output section
    st.subheader("Result")
    if uploaded_file is not None and st.session_state.get(
        f"predicted_{option_name}", False
    ):
        # Placeholder for output display
        # This will be replaced with actual prediction output
        st.image(image, caption="Processed Image (Placeholder)", use_column_width=True)
        st.text("Prediction details will appear here.")
    else:
        st.info("Upload an image and click 'Process' to see results.")


# Function to display video upload and processing section
def display_video_section():
    st.title("Option D - Video Processing")

    # File uploader for videos
    uploaded_file = st.file_uploader(
        "Upload a video for processing", type=["mp4", "avi", "mov"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            # Display the uploaded video
            st.video(tfile.name)

            # Process button
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    # Placeholder for video processing
                    # You will implement the actual processing here
                    st.success("Video processing completed!")
                    st.session_state["video_processed"] = True

    with col2:
        # Display output section
        st.subheader("Processing Output")
        if uploaded_file is not None and st.session_state.get("video_processed", False):
            # Placeholder for output video display
            # This will be replaced with actual processed video
            st.write("Processed video will appear here.")
            st.video(tfile.name)  # Replace with actual processed video
            st.text("Processing details will appear here.")
        else:
            st.info("Upload a video and click 'Process' to see results.")


def draw_bounding_boxes(image_path, inference_results):
    # Open image using PIL to get the size for drawing labels
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Convert the image to a numpy array for OpenCV to process
    image_cv = np.array(image)

    # Predefined colors for each class (you can expand this list)
    class_colors = {
        "bus": (0, 255, 0),  # Green for bus
        "person": (0, 0, 255),  # Blue for person
        # You can add more classes and their colors here.
    }

    # Function to generate a random color
    def random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Loop through the inference results to draw boxes and labels
    for result in inference_results:
        name = result["name"]
        confidence = result["confidence"]
        box = result["box"]

        # Get coordinates of the bounding box
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

        # Assign a color for the class or use random color if not in the predefined set
        color = class_colors.get(name, random_color())

        # Draw the bounding box with the selected color
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)  # Box thickness = 2

        # Label to display (name + confidence)
        label = f"{name} ({confidence*100:.2f}%)"
        print(label)

        # Draw the label text
        font = ImageFont.load_default()  # You can use any font if needed
        text_width, text_height = draw.textsize(label, font=font)
        # Draw a background rectangle for text to improve visibility
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill=color)
        draw.text(
            (x1, y1 - text_height), label, fill=(255, 255, 255), font=font
        )  # White text

    # Convert the image back to PIL format
    image = Image.fromarray(image_cv)
    # Image.Image.show(image)  # Display the image with bounding boxes

    return image
