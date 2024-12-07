"""Streamlit web application for image enhancement."""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import time
import logging
import sys
import pandas as pd
from typing import Optional, Tuple
import warnings
import torch

# Suppress all warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.utils.image_processor import ImageProcessor, EnhancementStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.processor = ImageProcessor()
        st.session_state.enhancement_attempts = []
        st.session_state.current_attempt = 0
        st.session_state.total_attempts = 5
        st.session_state.thresholds = {
            'sharpness': 0.95,  # Increased for maximum detail
            'contrast': 0.90,   # Optimized for dynamic range
            'noise': 0.15,      # Lowered to preserve fine details
            'detail': 0.95,     # Increased for enhanced clarity
            'color': 0.85       # Balanced for natural look
        }

def load_image(image_file) -> Optional[Image.Image]:
    """Load and prepare image for processing."""
    if image_file is None:
        return None
    
    try:
        # Read image file
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def process_image(image: Image.Image, strategy: EnhancementStrategy) -> Tuple[Image.Image, dict]:
    """Process image with multiple enhancement attempts."""
    try:
        # Reset enhancement attempts
        st.session_state.enhancement_attempts = []
        st.session_state.current_attempt = 0
        
        # Create progress containers
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        current_metrics_placeholder = st.empty()
        
        # Calculate original metrics
        metrics_original = st.session_state.processor.calculate_metrics(image)
        
        # Enhanced parameters for 5K resolution
        enhancement_params = [
            {'contrast': 1.4, 'sharpness': 1.5, 'color': 1.2},  # Maximum quality
            {'contrast': 1.5, 'sharpness': 1.6, 'color': 1.3},  # Ultra sharp
            {'contrast': 1.6, 'sharpness': 1.7, 'color': 1.25}, # Maximum detail
            {'contrast': 1.45, 'sharpness': 1.55, 'color': 1.2}, # Balanced
            {'contrast': 1.55, 'sharpness': 1.65, 'color': 1.3}  # Enhanced
        ]
        
        best_score = -float('inf')
        best_image = None
        best_metrics = None
        
        # Try different enhancement parameters
        for i, params in enumerate(enhancement_params):
            st.session_state.current_attempt = i + 1
            
            # Update progress
            progress = (i + 1) / len(enhancement_params)
            progress_bar = progress_placeholder.progress(progress)
            status_placeholder.text(f"Enhancement attempt {i+1}/{len(enhancement_params)} - Optimizing for 5K")
            
            # Process image with current parameters
            enhanced = st.session_state.processor.enhance_image(
                image, 
                strategy,
                enhancement_params=params
            )
            
            # Calculate metrics
            metrics = st.session_state.processor.calculate_metrics(enhanced)
            
            # Display current metrics
            with current_metrics_placeholder:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Current Enhancement Parameters")
                    st.write(f"Contrast: {params['contrast']:.2f}")
                    st.write(f"Sharpness: {params['sharpness']:.2f}")
                    st.write(f"Color: {params['color']:.2f}")
                with col2:
                    st.markdown("### Current Metrics")
                    st.write(f"Sharpness: {metrics['sharpness']:.2f}")
                    st.write(f"Contrast: {metrics['contrast']:.2f}")
                    st.write(f"Detail: {metrics['detail']:.2f}")
            
            # Calculate overall score with emphasis on 5K quality
            score = (
                metrics['sharpness'] * 0.35 +  # Increased weight for sharpness
                metrics['contrast'] * 0.25 +   # Balanced contrast
                (1 - metrics['noise_level']) * 0.15 +  # Reduced noise importance
                metrics['detail'] * 0.15 +     # Enhanced detail weight
                metrics['color'] * 0.10        # Natural color
            )
            
            # Update best result if better
            if score > best_score:
                best_score = score
                best_image = enhanced
                best_metrics = metrics
            
            # Record attempt
            st.session_state.enhancement_attempts.append({
                'attempt': i + 1,
                'parameters': params,
                'metrics': metrics,
                'score': score
            })
            
            # Small delay to show progress
            time.sleep(0.5)
        
        # Clear progress indicators
        progress_placeholder.empty()
        status_placeholder.empty()
        current_metrics_placeholder.empty()
        
        return best_image, {
            'original': metrics_original,
            'enhanced': best_metrics
        }
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return image, {}

def display_metrics(metrics: dict):
    """Display image quality metrics."""
    if not metrics:
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Original Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['✨ Sharpness', '🎯 Contrast', '🔍 Detail', '🎨 Color', '📱 Resolution'],
            'Value': [
                f"{metrics['original']['sharpness']:.2f}",
                f"{metrics['original']['contrast']:.2f}",
                f"{metrics['original']['detail']:.2f}",
                f"{metrics['original']['color']:.2f}",
                "Original"
            ]
        })
        st.dataframe(metrics_df, hide_index=True)
        
    with col2:
        st.markdown("### 📈 Enhanced Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['✨ Sharpness', '🎯 Contrast', '🔍 Detail', '🎨 Color', '📱 Resolution'],
            'Value': [
                f"{metrics['enhanced']['sharpness']:.2f}",
                f"{metrics['enhanced']['contrast']:.2f}",
                f"{metrics['enhanced']['detail']:.2f}",
                f"{metrics['enhanced']['color']:.2f}",
                "5K (5120x2880)"
            ]
        })
        st.dataframe(metrics_df, hide_index=True)

def display_enhancement_history():
    """Display enhancement attempt history."""
    if st.session_state.enhancement_attempts:
        st.markdown("### 📝 Enhancement History")
        history_df = pd.DataFrame([
            {
                'Attempt': attempt['attempt'],
                'Contrast': f"{attempt['parameters']['contrast']:.2f}",
                'Sharpness': f"{attempt['parameters']['sharpness']:.2f}",
                'Color': f"{attempt['parameters']['color']:.2f}",
                'Score': f"{attempt['score']:.2f}"
            }
            for attempt in st.session_state.enhancement_attempts
        ])
        st.dataframe(history_df, hide_index=True)

def main():
    """Main application function."""
    st.set_page_config(
        page_title="AI Image Enhancer",
        page_icon="🖼️",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    st.title("🖼️ AI Image Enhancer")
    st.markdown("""
    Enhance your images to 5K resolution with advanced AI-powered techniques. Upload an image and let
    our system optimize it with multiple enhancement attempts to achieve the perfect high-resolution result.
    """)
    
    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    
    # Quality thresholds
    st.sidebar.subheader("Quality Thresholds")
    
    # Update thresholds only if changed
    new_thresholds = {
        'sharpness': st.sidebar.slider("Sharpness Threshold", 0.0, 1.0, st.session_state.thresholds['sharpness'], 0.01),
        'contrast': st.sidebar.slider("Contrast Threshold", 0.0, 1.0, st.session_state.thresholds['contrast'], 0.01),
        'noise': st.sidebar.slider("Noise Threshold", 0.0, 1.0, st.session_state.thresholds['noise'], 0.01),
        'detail': st.sidebar.slider("Detail Threshold", 0.0, 1.0, st.session_state.thresholds['detail'], 0.01),
        'color': st.sidebar.slider("Color Threshold", 0.0, 1.0, st.session_state.thresholds['color'], 0.01)
    }
    
    # Only update if thresholds changed
    if new_thresholds != st.session_state.thresholds:
        st.session_state.thresholds = new_thresholds
        st.session_state.processor.update_thresholds(**new_thresholds)
    
    # Enhancement strategy selection
    st.sidebar.subheader("Enhancement Strategy")
    strategy_name = st.sidebar.selectbox(
        "Select Strategy",
        ["Auto"] + [s.value for s in EnhancementStrategy],
        index=0
    )
    
    strategy = EnhancementStrategy.AUTO if strategy_name == "Auto" else EnhancementStrategy(strategy_name)
    
    # Model information
    st.sidebar.subheader("🤖 AI Model Status")
    if torch.cuda.is_available():
        st.sidebar.success("GPU Acceleration: Enabled")
    else:
        st.sidebar.warning("GPU Acceleration: Disabled (Using CPU)")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        # Load and display original image
        image = load_image(uploaded_file)
        
        if image:
            # Process image
            with st.spinner("Enhancing image to 5K resolution..."):
                enhanced_image, metrics = process_image(image, strategy)
            
            # Display images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)
                
            with col2:
                st.markdown("### Enhanced Image (5K)")
                st.image(enhanced_image, use_container_width=True)
                
            # Display metrics
            display_metrics(metrics)
            
            # Display enhancement history
            display_enhancement_history()
            
            # Download button for enhanced image
            buf = io.BytesIO()
            enhanced_image.save(buf, format='PNG', quality=100)
            st.download_button(
                label="Download Enhanced 5K Image",
                data=buf.getvalue(),
                file_name="enhanced_5k_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
