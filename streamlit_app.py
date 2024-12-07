"""Streamlit app for image enhancement."""

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.utils.core.processor import Processor
from src.components.user_interface import ProgressUI
from src.utils.quality_management.quality_manager import QualityManager

# Page config
st.set_page_config(
    page_title="AI Image Enhancer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_image(image_file: Optional[bytes]) -> Optional[np.ndarray]:
    """Load and preprocess image.
    
    Args:
        image_file: Uploaded image file
        
    Returns:
        Preprocessed image array
    """
    if image_file is None:
        return None
        
    image = Image.open(image_file)
    image = np.array(image)
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

@st.cache_resource
def get_processor() -> Processor:
    """Get or create processor instance.
    
    Returns:
        Processor instance
    """
    return Processor()

def process_image(
    image: np.ndarray,
    quality: float,
    resolution: str,
    progress_ui: ProgressUI
) -> Tuple[np.ndarray, dict]:
    """Process image with progress tracking.
    
    Args:
        image: Input image array
        quality: Target quality score
        resolution: Target resolution
        progress_ui: Progress UI instance
        
    Returns:
        Tuple of (processed image, processing info)
    """
    processor = get_processor()
    
    # Create temp files
    input_path = "temp_input.png"
    output_path = "temp_output.png"
    cv2.imwrite(input_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    try:
        # Process image
        result = processor.process_image(
            input_path,
            output_path,
            target_quality=quality,
            target_resolution=resolution,
            progress_callback=progress_ui.update_progress
        )
        
        # Load result
        processed_image = cv2.imread(output_path)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        return processed_image, result
        
    finally:
        # Cleanup
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.remove(path)

def main():
    """Main application entry point."""
    st.title("✨ AI Image Enhancer")
    st.markdown("""
        Enhance your images with AI-powered features and real-time quality feedback.
        Upload an image and adjust the settings to get started!
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    quality = st.sidebar.slider(
        "Target Quality",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="Higher values produce better quality but take longer",
    )
    
    resolution = st.sidebar.selectbox(
        "Target Resolution",
        options=["4k", "5k", "8k"],
        index=1,
        help="Higher resolutions require more GPU memory",
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        help="Upload an image to enhance",
    )
    
    if uploaded_file:
        # Load and display input image
        image = load_image(uploaded_file.read())
        if image is not None:
            st.subheader("Input Image")
            st.image(image, use_column_width=True)
            
            # Process button
            if st.button("✨ Enhance Image"):
                # Initialize progress
                progress_ui = ProgressUI()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Process image
                    processed_image, info = process_image(
                        image,
                        quality,
                        resolution,
                        progress_ui
                    )
                    
                    # Display results
                    st.subheader("Enhanced Image")
                    st.image(processed_image, use_column_width=True)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Quality Score", f"{info['quality_score']:.2f}")
                    with col2:
                        st.metric("Processing Time", f"{info['duration']:.1f}s")
                    with col3:
                        st.metric("Resolution", resolution.upper())
                    
                    # Display detailed metrics
                    with st.expander("View Detailed Metrics"):
                        st.json(info['metrics'])
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    
                finally:
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ by Your Team | "
        "[GitHub](https://github.com/yourusername/image-enhancer) | "
        "[Documentation](https://github.com/yourusername/image-enhancer/docs)"
    )

if __name__ == "__main__":
    main()
