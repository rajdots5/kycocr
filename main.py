from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import uuid
import logging
from ocr_service import KYCService

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KYC_API")

app = FastAPI(
    title="KYC AI Extraction Service",
    description="API to extract structured details from Form ISR-1 images using PaddleOCR."
)

# Enable CORS (Cross-Origin Resource Sharing)
# This allows your frontend (static/index.html) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the core OCR service
# This is done at startup to keep the models loaded in memory for speed
kyc_service = KYCService()

# Ensure a temporary directory exists for uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract")
async def extract_kyc(file: UploadFile = File(...)):
    """
    ENDPOINT: Upload an image and receive structured JSON.
    1. Saves the uploaded file temporarily.
    2. Runs the Gatekeeper validation.
    3. Performs OCR and mapping.
    """
    logger.info(f"Received upload request: {file.filename}")

    # Validate file type (Allow images and PDF)
    content_type = file.content_type.lower()
    filename = file.filename.lower()
    
    is_image = "image" in content_type or filename.endswith(('.jpg', '.jpeg', '.png'))
    is_pdf = "pdf" in content_type or filename.endswith('.pdf')

    if not (is_image or is_pdf):
        logger.warning(f"Rejected invalid file: {filename} ({content_type})")
        raise HTTPException(status_code=400, detail="Please upload a valid Image or PDF file.")

    # Generate a unique filename to avoid collisions
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    try:
        # Save the file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to {file_path}")

        # Core Processing Loop
        try:
            # Step 1: Extract and Validate
            blocks = kyc_service.extract_raw_blocks(file_path)
            
            # Step 2: Map to JSON
            structured_result = kyc_service.process_form(blocks)
            
            logger.info("Extraction successful. Returning both raw and structured data.")
            return {
                "structured": structured_result,
                "raw": blocks
            }

        except ValueError as ve:
            # Handle the 'Gatekeeper' rejections (Quality/Form mismatch)
            error_msg = str(ve)
            if "IMAGE_REJECTED" in error_msg:
                clean_msg = error_msg.replace("IMAGE_REJECTED: ", "")
                logger.warning(f"Validation failed: {clean_msg}")
                raise HTTPException(status_code=400, detail=clean_msg)
            raise ve
            
    except HTTPException as he:
        # Re-raise HTTP exceptions (like 400 Bad Request)
        raise he
    except Exception as e:
        # Catch unexpected crashes and return 500
        logger.error(f"Unexpected system error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during processing.")
    
    finally:
        # Cleanup: Remove the temp file after processing to save space
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Temporary file {file_path} removed.")

# --- STATIC CONTENT ---
# Serve the 'static' folder on the root URL (/)
# This makes the frontend accessible at http://localhost:8000
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    logger.info("Starting KYC API Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
