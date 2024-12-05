"""Main Streamlit application module"""

import streamlit as st
from src.components.file_uploader import FileUploader
from src.utils.image_processor import ImageEnhancer
from PIL import Image
import io
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    st.title("Advanced Image Enhancer")
    st.write("Upload an image to enhance using AI-powered 4x Super Resolution")

    # Initialize components
    file_uploader = FileUploader()
    image_enhancer = ImageEnhancer()

    # Display model information
    st.sidebar.subheader("AI Model Information")
    st.sidebar.info(
        f"""
    **Current Model: {image_enhancer.get_model_name()}**
    
    **Model Capabilities:**
    - 4x Super Resolution upscaling
    - Enhanced detail preservation
    - Advanced noise reduction
    - Improved edge sharpness
    
    **Processing Device:** {image_enhancer.get_model_device()}
    
    **Note:** For optimal performance, the maximum output
    resolution is limited to 5K width. Larger images will
    be processed in sections for better quality.
    """
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display original image
            image = Image.open(uploaded_file)

            # Show original image details
            st.subheader("Source Image")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.info(
                    f"""
                **Original Image Details:**
                - Resolution: {image.size[0]}x{image.size[1]}px
                - Megapixels: {(image.size[0] * image.size[1] / 1000000):.2f}MP
                - Format: {image.format if image.format else 'Unknown'}
                - Color Mode: {image.mode}
                """
                )

            # Enhancement Settings
            st.subheader("Enhancement Settings")
            target_width = st.slider(
                "Target Resolution Width",
                min_value=max(1024, image.size[0]),  # Don't allow downscaling
                max_value=5120,  # Limited to 5K for better performance
                value=min(5120, max(1024, image.size[0] * 4)),  # Default to 4x or 5K
                step=256,
                help="Select output resolution width. Limited to 5K for optimal quality.",
            )

            # Show target output details
            target_height = int(target_width * (image.size[1] / image.size[0]))
            target_mp = (target_width * target_height) / 1000000

            st.info(
                f"""
            **Target Output Details:**
            - Resolution: {target_width}x{target_height}px
            - Target Megapixels: {target_mp:.2f}MP
            - Upscale Factor: {target_width/image.size[0]:.1f}x
            
            **Note:** Processing time may increase with larger output sizes.
            The model will automatically optimize memory usage for best quality.
            """
            )

            if st.button("Enhance Image"):
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                processing_info = st.empty()
                memory_info = st.empty()
                time_info = st.empty()

                try:
                    start_time = time.time()
                    last_update = start_time

                    def update_progress(progress: float, status: str):
                        """Update progress bar and status text with enhanced information"""
                        nonlocal last_update
                        current_time = time.time()
                        elapsed = current_time - start_time

                        # Update progress bar
                        progress_bar.progress(progress)

                        # Extract memory info if present
                        memory_stats = ""
                        if "GPU:" in status and "RAM:" in status:
                            memory_part = status[status.find("GPU:") :]
                            status = status[: status.find("GPU:")].strip()
                            memory_stats = f"**System Usage:**\n{memory_part}"

                        # Update status
                        status_text.markdown(f"**Status:** {status}")

                        if memory_stats:
                            memory_info.markdown(memory_stats)

                        # Show time information
                        time_info.markdown(
                            f"""
                        **Processing Time:**
                        - Elapsed: {elapsed:.1f}s
                        - Estimated Remaining: {(elapsed/progress - elapsed):.1f}s (if progress is linear)
                        """
                        )

                        if progress < 1.0:
                            processing_info.info(
                                """
                            💡 **Processing Status:**
                            - AI model is actively working
                            - Processing in small sections for stability
                            - Progress updates every few seconds
                            - Keep this tab open
                            """
                            )
                        else:
                            processing_info.success(
                                "✨ Enhancement completed successfully!"
                            )
                            memory_info.empty()
                            time_info.empty()

                        last_update = current_time

                    # Enhance image with progress updates
                    enhanced = image_enhancer.enhance_image(
                        image,
                        target_width=target_width,
                        progress_callback=update_progress,
                    )

                    # Calculate processing time
                    process_time = time.time() - start_time

                    # Clear progress indicators
                    status_text.empty()
                    progress_bar.empty()
                    processing_info.empty()

                    # Display enhanced image and details
                    st.subheader("Enhanced Image")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.image(
                            enhanced, caption="Enhanced Image", use_container_width=True
                        )
                    with col4:
                        st.success(
                            f"""
                        **Enhancement Complete!**
                        
                        **Enhanced Image Details:**
                        - Resolution: {enhanced.size[0]}x{enhanced.size[1]}px
                        - Megapixels: {(enhanced.size[0] * enhanced.size[1] / 1000000):.2f}MP
                        - Upscale Factor: {enhanced.size[0]/image.size[0]:.2f}x
                        - Processing Time: {process_time:.2f}s
                        """
                        )

                    # Enhancement Information
                    st.subheader("Enhancement Process Details")
                    st.info(
                        f"""
                    **Three-Stage Enhancement:**
                    1. **Pre-processing:**
                       - Advanced image preparation
                       - Memory optimization
                       - Color space normalization
                    
                    2. **AI Super Resolution:**
                       - EDSR 4x model processing
                       - Deep feature extraction
                       - Detail reconstruction
                       - Noise suppression
                    
                    3. **Post-processing:**
                       - High-quality upscaling to {target_width}px width
                       - Color accuracy refinement
                       - Final detail enhancement
                    
                    **Resolution Improvement:**
                    - Input: {image.size[0]}x{image.size[1]}px ({image.size[0] * image.size[1]:,} pixels)
                    - Output: {enhanced.size[0]}x{enhanced.size[1]}px ({enhanced.size[0] * enhanced.size[1]:,} pixels)
                    - Total Pixels Added: {(enhanced.size[0] * enhanced.size[1]) - (image.size[0] * image.size[1]):,}
                    """
                    )

                    # Convert to bytes for download
                    buf = io.BytesIO()
                    enhanced.save(buf, format="PNG")
                    buf.seek(0)

                    # Add download button
                    st.download_button(
                        label="Download Enhanced Image",
                        data=buf,
                        file_name="enhanced_image.png",
                        mime="image/png",
                    )

                except Exception as e:
                    st.error(
                        f"""
                    ❌ **Error enhancing image:** {str(e)}
                    
                    **Possible solutions:**
                    - Try reducing the target resolution
                    - Use a smaller input image
                    - Check if your system has enough memory
                    - Wait a few minutes and try again
                    """
                    )
                    logger.error(f"Enhancement error: {str(e)}", exc_info=True)

                    # Clear progress indicators on error
                    status_text.empty()
                    progress_bar.empty()
                    processing_info.empty()
                    memory_info.empty()
                    time_info.empty()

        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            logger.error(f"Image loading error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
