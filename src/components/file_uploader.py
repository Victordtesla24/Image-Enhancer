"""File uploader component"""

from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUploader:
    """Handle file uploads and validation"""

    def validate_image(self, file) -> Image.Image:
        """
        Validate and load an image file from Streamlit's uploaded file

        Args:
            file: UploadedFile from Streamlit's file_uploader

        Returns:
            PIL Image object

        Raises:
            ValueError: If file is invalid
        """
        try:
            logger.info(f"Starting image validation for uploaded file")

            if not file:
                logger.error("No file provided")
                raise ValueError("No file provided")

            # Read file content
            logger.info("Reading file content...")
            content = file.getvalue()
            if not content:
                logger.error("Empty file provided")
                raise ValueError("Empty file provided")

            # Try to open as image
            logger.info("Attempting to open file as image...")
            try:
                image = Image.open(io.BytesIO(content))
                image.verify()  # Verify it's actually an image
                logger.info(f"Image verified successfully - Format: {image.format}")
            except Exception as e:
                logger.error(f"Image verification failed: {str(e)}")
                raise ValueError(f"Invalid image format: {str(e)}")

            # Re-open because verify() closes the file
            image = Image.open(io.BytesIO(content))
            logger.info(
                f"Image loaded successfully - Size: {image.size}, Mode: {image.mode}"
            )

            return image

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during image validation: {str(e)}")
            raise ValueError(f"Invalid image file: {str(e)}")
