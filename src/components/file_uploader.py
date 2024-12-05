"""File uploader component"""

from fastapi import UploadFile, HTTPException
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUploader:
    """Handle file uploads and validation"""

    def validate_image(self, file: UploadFile) -> Image.Image:
        """
        Validate and load an image file

        Args:
            file: File to validate

        Returns:
            PIL Image object

        Raises:
            HTTPException: If file is invalid
        """
        try:
            logger.info(f"Starting image validation for file: {file.filename}")

            if not file:
                logger.error("No file provided")
                raise HTTPException(status_code=400, detail="No file provided")

            if not file.content_type or not file.content_type.startswith("image/"):
                logger.error(f"Invalid content type: {file.content_type}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid content type: {file.content_type}. Expected image/*",
                )

            # Read file content
            logger.info("Reading file content...")
            content = file.file.read()
            if not content:
                logger.error("Empty file provided")
                raise HTTPException(status_code=400, detail="Empty file provided")

            # Try to open as image
            logger.info("Attempting to open file as image...")
            try:
                image = Image.open(io.BytesIO(content))
                image.verify()  # Verify it's actually an image
                logger.info(f"Image verified successfully - Format: {image.format}")
            except Exception as e:
                logger.error(f"Image verification failed: {str(e)}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid image format: {str(e)}"
                )

            # Re-open because verify() closes the file
            image = Image.open(io.BytesIO(content))
            logger.info(
                f"Image loaded successfully - Size: {image.size}, Mode: {image.mode}"
            )

            # Reset file pointer for potential future reads
            file.file.seek(0)

            return image

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during image validation: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
