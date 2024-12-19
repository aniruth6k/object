import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from deepface import DeepFace
from supabase import create_client, Client
import datetime
from typing import Dict, List
from pydantic import BaseModel, validator
import base64
from geopy.geocoders import Nominatim
import tensorflow as tf
import tempfile
import logging
import shutil
import traceback
from PIL import Image
import io

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

class LocationData(BaseModel):
    latitude: float
    longitude: float
    image: str
    is_checkout: bool = False
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Invalid latitude value')
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Invalid longitude value')
        return v
    
    @validator('image')
    def validate_image(cls, v):
        try:
            base64.b64decode(v)
        except:
            raise ValueError('Invalid base64 image')
        return v

class Response(BaseModel):
    success: bool
    message: str = ""
    data: Dict = {}

class FacultyData(BaseModel):
    faculty_id: str
    name: str
    department: str
    image: str

class AttendanceHistory(BaseModel):
    faculty_id: str
    start_date: str
    end_date: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL="https://ohzkpmpwzfklznxgficm.supabase.co"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9oemtwbXB3emZrbHpueGdmaWNtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ0NTYxNzAsImV4cCI6MjA1MDAzMjE3MH0.9JMQzYW3CGfPEV6Abp_Cm7tJfVnLgs9Xz6n4ArYUVmU"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

geolocator = Nominatim(user_agent="faculty_attendance_app")

def initialize_db():
    if not os.path.exists("faculty_photos_db"):
        os.makedirs("faculty_photos_db")
        logger.info("Created faculty_photos_db directory")

def verify_face_in_image(image_path: str) -> bool:
    try:
        # Try multiple face detection backends in order of reliability
        backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
        
        for backend in backends:
            try:
                faces = DeepFace.extract_faces(
                    img_path=image_path,
                    enforce_detection=True,
                    detector_backend=backend
                )
                if len(faces) > 0:
                    # Verify face quality
                    img = cv2.imread(image_path)
                    face = faces[0]
                    face_region = img[
                        face['facial_area']['y']:face['facial_area']['y'] + face['facial_area']['h'],
                        face['facial_area']['x']:face['facial_area']['x'] + face['facial_area']['w']
                    ]
                    
                    # Check face size
                    if face['facial_area']['w'] < 100 or face['facial_area']['h'] < 100:
                        continue
                        
                    # Check face confidence
                    if face.get('confidence', 0) < 0.9:
                        continue
                        
                    # Check for blur
                    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if blur_value < 100:  # Adjust threshold as needed
                        continue
                        
                    return True
            except:
                continue
                
        return False
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        return False

def preprocess_image(image_path: str) -> str:
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return image_path
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Enhance edges
        edges = cv2.Canny(denoised, 50, 150)
        enhanced_edges = cv2.addWeighted(denoised, 0.7, edges, 0.3, 0)
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_edges, cv2.COLOR_GRAY2BGR)
        
        # Adjust brightness and contrast
        alpha = 1.2  # Contrast
        beta = 10    # Brightness
        adjusted = cv2.convertScaleAbs(enhanced_bgr, alpha=alpha, beta=beta)
        
        # Save preprocessed image
        preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg')
        cv2.imwrite(preprocessed_path, adjusted)
        
        return preprocessed_path
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return image_path

def find_matching_face(img_path: str, confidence_threshold: float = 0.6) -> tuple:
    """Returns (faculty_id, confidence) if match found, else (None, 0)"""
    try:
        # Try multiple recognition models
        models = ['VGG-Face', 'Facenet', 'OpenFace']
        metrics = ['cosine', 'euclidean']
        
        best_match = None
        best_confidence = 0
        
        for model in models:
            for metric in metrics:
                try:
                    result = DeepFace.find(
                        img_path=img_path,
                        db_path="faculty_photos_db",
                        enforce_detection=False,
                        model_name=model,
                        distance_metric=metric,
                        detector_backend='retinaface'
                    )
                    
                    if isinstance(result, list) and len(result) > 0 and not result[0].empty:
                        df_result = result[0]
                        df_result = df_result.sort_values('distance')
                        match = df_result.iloc[0]
                        
                        confidence = 1 - float(match['distance'])
                        if confidence > best_confidence:
                            identity_path = match['identity']
                            faculty_id = os.path.splitext(os.path.basename(identity_path))[0]
                            best_match = faculty_id
                            best_confidence = confidence
                            
                except Exception as e:
                    logger.error(f"Error with model {model}, metric {metric}: {str(e)}")
                    continue
                    
        if best_confidence > confidence_threshold:
            return best_match, best_confidence
            
        return None, 0
        
    except Exception as e:
        logger.error(f"Face matching failed: {str(e)}")
        return None, 0

@app.post("/faculty_attendance", response_model=Response)
async def faculty_attendance(location_data: LocationData):
    temp_file_path = None
    preprocessed_path = None
    try:
        # Validate and save image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image_data = base64.b64decode(location_data.image)
            temp_file.write(image_data)
            temp_file_path = temp_file.name
            
        # Enhance image quality
        preprocessed_path = preprocess_image(temp_file_path)
        
        # Verify face presence and quality
        if not verify_face_in_image(preprocessed_path):
            return Response(
                success=False,
                message="No clear face detected. Please ensure good lighting and try again.",
                data={}
            )
            
        # Get location info
        try:
            location = geolocator.reverse(f"{location_data.latitude}, {location_data.longitude}")
            address = location.address if location else "Unknown Location"
        except Exception as e:
            logger.error(f"Location lookup failed: {str(e)}")
            address = "Location lookup failed"
            
        # Find matching faculty
        faculty_id, confidence = find_matching_face(preprocessed_path)
        
        if faculty_id:
            faculty_info_query = supabase.table('faculty_info').select("*").eq('faculty_id', faculty_id).execute()
            
            if not faculty_info_query.data:
                return Response(
                    success=False,
                    message="Faculty information not found in database",
                    data={}
                )
                
            faculty_info = faculty_info_query.data[0]
            current_time = datetime.datetime.now()
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            attendance_data = {
                'faculty_id': faculty_id,
                'name': faculty_info['name'],
                'department': faculty_info['department'],
                'latitude': location_data.latitude,
                'longitude': location_data.longitude,
                'location_address': address,
                'confidence': confidence
            }
            
            existing_attendance = supabase.table('faculty_attendance')\
                .select("*")\
                .eq('faculty_id', faculty_id)\
                .gte('check_in_time', today_start.isoformat())\
                .is_('check_out_time', 'null')\
                .execute()
                
            if location_data.is_checkout:
                if not existing_attendance.data:
                    return Response(
                        success=False,
                        message="No active check-in found. Please check-in first.",
                        data={}
                    )
                    
                update_data = {
                    'check_out_time': current_time.isoformat(),
                    'check_out_latitude': location_data.latitude,
                    'check_out_longitude': location_data.longitude,
                    'check_out_address': address
                }
                
                supabase.table('faculty_attendance')\
                    .update(update_data)\
                    .eq('id', existing_attendance.data[0]['id'])\
                    .execute()
                    
                return Response(
                    success=True,
                    message=f"Goodbye {faculty_info['name']}! Check-out recorded successfully.",
                    data=attendance_data
                )
            else:
                if existing_attendance.data:
                    return Response(
                        success=False,
                        message="Already checked in. Please check-out first.",
                        data={}
                    )
                    
                attendance_data['check_in_time'] = current_time.isoformat()
                supabase.table('faculty_attendance').insert(attendance_data).execute()
                
                return Response(
                    success=True,
                    message=f"Welcome {faculty_info['name']}! Check-in recorded successfully.",
                    data=attendance_data
                )
                
        return Response(
            success=False,
            message="No matching faculty found. Please try again or contact administrator.",
            data={}
        )
        
    except Exception as e:
        logger.error(f"Attendance error: {traceback.format_exc()}")
        return Response(
            success=False,
            message=str(e),
            data={}
        )
    finally:
        # Clean up temporary files
        for path in [temp_file_path, preprocessed_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.post("/register_faculty", response_model=Response)
async def register_faculty(faculty_data: FacultyData):
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image_data = base64.b64decode(faculty_data.image)
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        if not verify_face_in_image(temp_file_path):
            os.unlink(temp_file_path)
            return Response(
                success=False,
                message="No clear face detected in image. Please try again.",
                data={}
            )

        faculty_id = faculty_data.faculty_id
        photo_path = f"faculty_photos_db/{faculty_id}.jpg"
        
        shutil.move(temp_file_path, photo_path)
        
        faculty_info = {
            'faculty_id': faculty_id,
            'name': faculty_data.name,
            'department': faculty_data.department,
            'photo_path': photo_path
        }
        
        result = supabase.table('faculty_info').insert(faculty_info).execute()
        
        try:
            DeepFace.find(
                img_path=photo_path,
                db_path="faculty_photos_db",
                enforce_detection=False,
                model_name="VGG-Face",
            )
            logger.info(f"Registered new faculty: {faculty_id}")
        except Exception as e:
            logger.error(f"Error initializing face database: {str(e)}")
        
        return Response(
            success=True,
            message="Faculty registered successfully",
            data=faculty_info
        )
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return Response(
            success=False,
            message=f"Registration failed: {str(e)}",
            data={}
        )

@app.post("/get_attendance_history", response_model=Response)
async def get_attendance_history(request: AttendanceHistory):
    try:
        attendance_records = supabase.table('faculty_attendance')\
            .select("*")\
            .eq('faculty_id', request.faculty_id)\
            .gte('check_in_time', request.start_date)\
            .lte('check_in_time', request.end_date)\
            .execute()

        if not attendance_records.data:
            return Response(
                success=True,
                message="No attendance records found for the specified period",
                data={'records': []}
            )

        formatted_records = []
        for record in attendance_records.data:
            formatted_record = {
                'date': record['check_in_time'].split('T')[0],
                'check_in_time': record['check_in_time'],
                'check_in_location': record['location_address'],
                'check_out_time': record['check_out_time'],
                'check_out_location': record.get('check_out_address', 'Not checked out'),
                'department': record['department'],
                'name': record['name']
            }
            formatted_records.append(formatted_record)

        return Response(
            success=True,
            message="Attendance history retrieved successfully",
            data={'records': formatted_records}
        )

    except Exception as e:
        logger.error(f"Error retrieving attendance history: {str(e)}")
        return Response(
            success=False,
            message=f"Error retrieving attendance history: {str(e)}",
            data={}
        )

@app.on_event("startup")
async def startup_event():
    initialize_db()

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
