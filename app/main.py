"""
FastAPI service for the image captioning model.

Run locally with:

    uvicorn app.main:app --reload --port 8000

API:
    http://127.0.0.1:8000

Swagger:
    http://127.0.0.1:8000/docs

Frontend:
    frontend/index.html
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, UnidentifiedImageError

from app.inference import CaptionModel


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Image Captioning API",
    description="Upload an image and get a generated caption back.",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

# Allow the local frontend to communicate with FastAPI.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load model ONCE
# --------------------------------------------------

# The model is loaded when FastAPI starts.
#
# We DON'T load the model inside /caption because
# that would reload the model for every image request.

caption_model = CaptionModel()


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def health_check() -> dict:
    """
    Check whether the API is running.
    """

    return {
        "status": "ok",
        "message": "Image captioning API is running."
    }


# --------------------------------------------------
# Caption endpoint
# --------------------------------------------------

@app.post("/caption")
async def generate_caption(
    file: UploadFile = File(...)
) -> dict:
    """
    Accept an uploaded image and generate a caption.
    """

    # --------------------------------------------------
    # 1. Check that uploaded file is an image
    # --------------------------------------------------

    if (
        file.content_type is None
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file must be an image, "
                f"got content_type={file.content_type!r}"
            ),
        )


    # --------------------------------------------------
    # 2. Read the uploaded image
    # --------------------------------------------------

    try:

        raw_bytes = await file.read()

        image = Image.open(
            io.BytesIO(raw_bytes)
        ).convert("RGB")

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded file as an image. "
                "It may be corrupted."
            ),
        )

    except OSError as error:

        raise HTTPException(
            status_code=400,
            detail=f"Error loading uploaded image: {error}",
        )


    # --------------------------------------------------
    # 3. Generate caption
    # --------------------------------------------------

    try:

        caption = caption_model.caption_image(image)

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Caption generation failed: {error}",
        )


    # --------------------------------------------------
    # 4. Return result
    # --------------------------------------------------

    return {
        "filename": file.filename,
        "caption": caption,
    }