"""Main application module"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from src.components.file_uploader import FileUploader
from src.utils.image_processor import ImageEnhancer
import io
import logging
from PIL import Image
import traceback

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
async def enhance_image(request: Request, target_width: int = 7680):
    """
    Enhance an image

    Args:
        request: Request object containing the image file in body
        target_width: Desired width of output image (default: 7680, max: 7680)

    Returns:
        Enhanced image as PNG

    Raises:
        HTTPException: If input is invalid or processing fails
    """
    try:
        # Read raw body
        logger.info("Reading request body...")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="No image data provided")

        logger.info(
            f"Received enhancement request - Content Length: {len(body)}, Target Width: {target_width}"
        )

        # Validate target width
        if target_width <= 0:
            logger.error(f"Invalid target width: {target_width}")
            raise HTTPException(status_code=400, detail="Target width must be positive")

        # Enforce maximum target width
        target_width = min(target_width, 7680)
        logger.info(f"Using target width: {target_width} (max: 7680)")

        # Load and validate image
        logger.info("Validating image...")
        try:
            image = Image.open(io.BytesIO(body))
            image.verify()  # Verify it's actually an image
            # Re-open because verify() closes the file
            image = Image.open(io.BytesIO(body))
            logger.info(
                f"Image validated successfully - Size: {image.size}, Mode: {image.mode}"
            )
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=400, detail=f"Invalid image format: {str(e)}"
            )

        # Enhance image
        try:
            logger.info("Starting image enhancement...")
            enhanced = image_enhancer.enhance_image(image, target_width=target_width)
            logger.info(f"Enhancement completed - New size: {enhanced.size}")
        except RuntimeError as e:
            logger.error(
                f"Runtime error during enhancement: {str(e)}\n{traceback.format_exc()}"
            )
            if "out of memory" in str(e).lower():
                raise HTTPException(
                    status_code=507,
                    detail="Insufficient memory to process image. Try reducing target width.",
                )
            raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")
        except Exception as e:
            logger.error(
                f"Error during enhancement: {str(e)}\n{traceback.format_exc()}"
            )
            raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

        # Convert to bytes
        logger.info("Converting enhanced image to PNG...")
        try:
            img_byte_arr = io.BytesIO()
            enhanced.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            result = img_byte_arr.getvalue()
            logger.info(f"Conversion complete - Output size: {len(result)} bytes")
        except Exception as e:
            logger.error(f"Error converting to PNG: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail=f"Failed to convert image: {str(e)}"
            )

        logger.info("Enhancement process completed successfully")
        return Response(content=result, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
