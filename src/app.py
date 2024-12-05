"""Main application module"""

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from src.components.file_uploader import FileUploader
from src.utils.image_processor import ImageEnhancer
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_uploader = FileUploader()
image_enhancer = ImageEnhancer()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("src/static/index.html") as f:
        return HTMLResponse(content=f.read())


@app.post("/enhance")
async def enhance_image(file: UploadFile, target_width: int = 5120):
    """
    Enhance an image

    Args:
        file: Image file to enhance
        target_width: Desired width of output image

    Returns:
        Enhanced image as PNG

    Raises:
        HTTPException: If input is invalid or processing fails
    """
    try:
        logger.info(
            f"Received enhancement request - File: {file.filename}, Content-Type: {file.content_type}, Target Width: {target_width}"
        )

        # Validate target width
        if target_width <= 0:
            logger.error(f"Invalid target width: {target_width}")
            raise HTTPException(status_code=400, detail="Target width must be positive")

        # Validate and load image
        logger.info("Validating image...")
        try:
            image = file_uploader.validate_image(file)
            logger.info(
                f"Image validated successfully - Size: {image.size}, Mode: {image.mode}"
            )
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            raise

        # Enhance image
        try:
            logger.info("Starting image enhancement...")
            enhanced = image_enhancer.enhance_image(image, target_width=target_width)
            logger.info(f"Enhancement completed - New size: {enhanced.size}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"Out of memory error during enhancement: {str(e)}")
                raise HTTPException(
                    status_code=507,
                    detail="Insufficient memory to process image. Try reducing target width.",
                )
            logger.error(f"Runtime error during enhancement: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error during enhancement: {str(e)}")
            raise

        # Convert to bytes
        logger.info("Converting enhanced image to PNG...")
        img_byte_arr = io.BytesIO()
        enhanced.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        logger.info("Enhancement process completed successfully")
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
