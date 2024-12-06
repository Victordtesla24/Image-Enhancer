"""Streamlit app for advanced image enhancement"""

import io
import logging
import streamlit as st
from PIL import Image
from src.components.file_uploader import FileUploader
from src.utils.image_processor import ImageEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for the Streamlit app"""
    st.set_page_config(page_title="5K AI Image Enhancer", page_icon="🖼️", layout="wide")

    st.title("🖼️ 5K AI Image Enhancer")
    st.markdown(
        """
    Enhance your images up to 5K resolution using cutting-edge AI technology. 
    Upload an image and see it transformed using multiple state-of-the-art enhancement models!
    """
    )

    # Initialize components
    if "image_enhancer" not in st.session_state:
        st.session_state.image_enhancer = ImageEnhancer()

    if "file_uploader" not in st.session_state:
        st.session_state.file_uploader = FileUploader()

    # Get available models
    available_models = st.session_state.image_enhancer.get_available_models()

    # Sidebar - Model Selection and Info
    st.sidebar.header("🤖 Enhancement Models")
    selected_models = []
    for model in available_models:
        if st.sidebar.checkbox(
            f"Use {model['name']}", value=True, key=f"model_{model['name']}"
        ):
            selected_models.append(model["name"])
        with st.sidebar.expander(f"ℹ️ About {model['name']}"):
            st.write(model["description"])

    # Enhancement settings
    st.sidebar.header("⚙️ Enhancement Settings")

    # Resolution presets
    resolution_presets = {
        "5K (5120x2880)": 5120,
        "4K (3840x2160)": 3840,
        "2K (2048x1080)": 2048,
        "Full HD (1920x1080)": 1920,
        "Custom": "custom",
    }

    selected_preset = st.sidebar.selectbox(
        "Resolution Preset",
        options=list(resolution_presets.keys()),
        index=0,  # Default to 5K
        help="Select target resolution preset or choose custom",
    )

    if selected_preset == "Custom":
        target_width = st.sidebar.number_input(
            "Custom Width",
            min_value=512,
            max_value=5120,
            value=1920,
            step=128,
            help="Enter custom target width (512-5120 pixels)",
        )
    else:
        target_width = resolution_presets[selected_preset]

    # Show selected resolution info
    st.sidebar.info(f"Target Width: {target_width}px")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"],
        help="Upload an image file to enhance",
    )

    # Process image when uploaded
    if uploaded_file is not None:
        try:
            # Read and validate image
            logger.info("Validating uploaded image")
            image_bytes = uploaded_file.getvalue()
            input_image = st.session_state.file_uploader.validate_image(
                io.BytesIO(image_bytes)
            )

            # Display original and enhanced images side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(input_image, use_container_width=True)
                st.info(f"Size: {input_image.size[0]}x{input_image.size[1]}px")

            # Add enhance button
            if st.button("🔄 Enhance Image", key="enhance_button"):
                if not selected_models:
                    st.error(
                        "Please select at least one enhancement model from the sidebar!"
                    )
                else:
                    with st.spinner("Processing image..."):
                        try:
                            # Create progress tracking
                            progress_container = st.empty()
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            def update_progress(progress, status):
                                """Update progress bar and status text"""
                                progress_bar.progress(progress)
                                status_text.text(status)
                                progress_container.text(
                                    f"Progress: {progress*100:.0f}%"
                                )

                            # Enhance image
                            logger.info("Starting image enhancement")
                            enhanced_image, enhancement_details = (
                                st.session_state.image_enhancer.enhance_image(
                                    input_image,
                                    target_width=target_width,
                                    models=selected_models,
                                    progress_callback=update_progress,
                                )
                            )

                            with col2:
                                st.subheader("Enhanced Image")
                                st.image(enhanced_image, use_container_width=True)
                                st.info(
                                    f"Size: {enhanced_image.size[0]}x{enhanced_image.size[1]}px"
                                )

                                # Enhancement details
                                with st.expander(
                                    "📊 Enhancement Details", expanded=True
                                ):
                                    st.markdown("**Source Image**")
                                    st.text(
                                        f"Size: {enhancement_details['source_size']}"
                                    )
                                    st.text(f"DPI: {enhancement_details['source_dpi']}")

                                    st.markdown("**Models Applied**")
                                    for model in enhancement_details["models_used"]:
                                        st.markdown(
                                            f"- **{model['name']}**: {model['description']}"
                                        )

                                    st.markdown("**Output Image**")
                                    st.text(
                                        f"Size: {enhancement_details['target_size']}"
                                    )
                                    st.text(f"DPI: {enhancement_details['target_dpi']}")
                                    st.text(
                                        f"Processing Time: {enhancement_details['processing_time']}"
                                    )

                                    # Quality Check Results
                                    st.markdown("**Quality Check Results**")
                                    quality_checks = enhancement_details[
                                        "quality_checks"
                                    ]
                                    for check, details in quality_checks.items():
                                        status = "✅" if details["passed"] else "❌"
                                        st.markdown(
                                            f"- **{check}**: {status}\n  "
                                            f"  - Current: {details['value']}\n  "
                                            f"  - Required: {details['required']}"
                                        )

                                # Add download button
                                buf = io.BytesIO()
                                enhanced_image.save(buf, format="PNG", quality=95)
                                byte_im = buf.getvalue()

                                st.download_button(
                                    label="📥 Download Enhanced Image",
                                    data=byte_im,
                                    file_name="enhanced_image.png",
                                    mime="image/png",
                                )

                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            progress_container.empty()
                            st.success("Enhancement complete! ✨")

                        except Exception as e:
                            st.error(f"Error during enhancement: {str(e)}")
                            logger.error(f"Enhancement error: {str(e)}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            logger.error(f"Processing error: {str(e)}")

    # Add usage instructions
    with st.expander("📖 How to Use"):
        st.markdown(
            """
        1. Select your desired output resolution from the presets or choose custom
        2. Select the enhancement models you want to use from the sidebar
        3. Upload an image using the file uploader above
        4. Click the 'Enhance Image' button
        5. View the enhancement details and quality check results
        6. Download the enhanced image if satisfied
        
        **Quality Requirements:**
        - Resolution: Minimum 5K (5120x2880)
        - Color Depth: 24-bit RGB or 32-bit RGBA
        - DPI: Minimum 300
        - Dynamic Range: 220-255
        - Sharpness: Minimum 70
        - Noise Level: Maximum 120
        - File Size: Minimum 1.5MB
        
        **Note:** 
        - Higher resolutions (4K/5K) may take longer to process
        - Multiple models can be combined for better results
        - For optimal results, try different combinations of models
        """
        )

    # Add footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Python, Streamlit, and advanced image processing")


if __name__ == "__main__":
    main()
