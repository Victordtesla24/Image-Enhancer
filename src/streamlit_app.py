"""Streamlit interface for image enhancement"""

import streamlit as st
from PIL import Image
import io
from utils.image_processor import ImageEnhancer
from config.settings import DEFAULT_TARGET_WIDTH, MAX_FILE_SIZE


def main():
    st.title("Image Enhancer")
    st.write("Upload an image to enhance its quality")

    # Initialize image enhancer
    image_enhancer = ImageEnhancer()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE:
            st.error(f"File size exceeds maximum limit of {MAX_FILE_SIZE}MB")
            return

        try:
            # Load and display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)

            # Add enhancement button
            if st.button("Enhance Image"):
                with st.spinner("Enhancing image..."):
                    try:
                        # Enhance image
                        enhanced_image, enhancement_details = (
                            image_enhancer.enhance_image(
                                image, target_width=DEFAULT_TARGET_WIDTH
                            )
                        )

                        # Display enhanced image
                        st.image(
                            enhanced_image,
                            caption="Enhanced Image",
                            use_container_width=True,
                        )

                        # Show enhancement details
                        st.write("Enhancement Details:", enhancement_details)

                        # Add download button
                        buf = io.BytesIO()
                        enhanced_image.save(buf, format="PNG")
                        st.download_button(
                            label="Download Enhanced Image",
                            data=buf.getvalue(),
                            file_name="enhanced_image.png",
                            mime="image/png",
                        )

                    except Exception as e:
                        st.error(f"Error enhancing image: {str(e)}")

        except Exception as e:
            st.error(f"Error loading image: {str(e)}")


if __name__ == "__main__":
    main()
