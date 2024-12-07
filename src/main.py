"""Main application entry point."""

import os
import sys
import logging
from pathlib import Path
import argparse

from utils.image_processor import ImageProcessor
from utils.core.gpu_accelerator import GPUAccelerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description='Image Enhancement Application')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('--output', '-o', help='Output directory (default: enhanced/)', default='enhanced')
    parser.add_argument('--batch-size', '-b', type=int, help='Batch size for processing', default=4)
    parser.add_argument('--quality', '-q', choices=['low', 'medium', 'high'], 
                       help='Enhancement quality level', default='medium')
    return parser

def process_image(processor: ImageProcessor, input_path: str, output_path: str) -> bool:
    """Process a single image.
    
    Args:
        processor: Image processor instance
        input_path: Path to input image
        output_path: Path to save enhanced image
        
    Returns:
        True if successful
    """
    try:
        # Load image
        image = processor.load_image(input_path)
        if image is None:
            logger.error(f"Failed to load image: {input_path}")
            return False
            
        # Enhance image
        enhanced = processor.enhance_image(image)
        
        # Save enhanced image
        if not processor.save_image(enhanced, output_path):
            logger.error(f"Failed to save enhanced image: {output_path}")
            return False
            
        logger.info(f"Successfully enhanced image: {input_path} -> {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing image {input_path}: {str(e)}")
        return False

def process_directory(processor: ImageProcessor, input_dir: str, output_dir: str) -> bool:
    """Process all images in a directory.
    
    Args:
        processor: Image processor instance
        input_dir: Input directory path
        output_dir: Output directory path
        
    Returns:
        True if all images processed successfully
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each image
        success = True
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"enhanced_{filename}")
                if not process_image(processor, input_path, output_path):
                    success = False
                    
        return success
    except Exception as e:
        logger.error(f"Error processing directory {input_dir}: {str(e)}")
        return False

def main():
    """Main application entry point."""
    try:
        # Parse command line arguments
        parser = setup_argparse()
        args = parser.parse_args()
        
        # Initialize components
        logger.info("Initializing components...")
        accelerator = GPUAccelerator()
        processor = ImageProcessor()
        
        # Configure processor
        processor.config.update({
            'batch_size': args.batch_size,
            'enhancement_level': args.quality
        })
        
        # Process input
        input_path = os.path.abspath(args.input)
        output_path = os.path.abspath(args.output)
        
        if os.path.isfile(input_path):
            # Process single image
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, f"enhanced_{os.path.basename(input_path)}")
            success = process_image(processor, input_path, output_path)
        elif os.path.isdir(input_path):
            # Process directory
            success = process_directory(processor, input_path, output_path)
        else:
            logger.error(f"Input path does not exist: {input_path}")
            return 1
            
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        return 1
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        if 'processor' in locals():
            processor.cleanup()
        if 'accelerator' in locals():
            accelerator.cleanup()

if __name__ == '__main__':
    sys.exit(main()) 