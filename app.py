import os
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from predictions import ModelPredictor

# Create required directories
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/predictions", exist_ok=True)

app = FastAPI(title="Tumor Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving uploaded images and predictions
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            response = {
                "file": False,
            }
            return JSONResponse(response, status_code=400)

        # Create unique filename
        file_extension = os.path.splitext(image.filename)[1]
        image_name = f"{uuid.uuid4()}{file_extension}"
        image_path = f"uploads/images/{image_name}"

        # Save uploaded file
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Process image with ModelPredictor
        predictor = ModelPredictor(image_path)

        try:
            predictor.predict()
            predictor.set_params()

            if predictor.detections is None:
                response = {
                    "predictions": False,
                }

                return JSONResponse(response, status_code=400)

            predictor.plot_image()

            # Prepare response
            response = {
                "predictions": True,
                "message": "Image processed successfully",
                "data": {
                    "detections": predictor.detections,
                    "original_image": f"/uploads/images/{image_name}",
                    "prediction_image": (
                        predictor.result_image_path.replace("uploads", "/uploads")
                        if predictor.result_image_path
                        else None
                    ),
                },
            }
            return JSONResponse(response, status_code=200)

        except Exception as e:

            response = {
                "predictions": False,
            }
            return JSONResponse(response, status_code=400)

    except Exception as e:

        response = {
            "upload": False,
        }
        return JSONResponse(response, status_code=400)
