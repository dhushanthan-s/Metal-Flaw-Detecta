from django.shortcuts import render
from django.http import HttpResponseBadRequest, HttpResponseForbidden, HttpResponseNotAllowed, JsonResponse, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import generics
from tensorflow.keras.models import load_model
from io import StringIO

import zipfile
import os
import cv2
import numpy as np

from .models import Users
from .serializers import UserSerializer
from ..model.defect_detector.defect_detection import DefectDetector
from ..model.defect_detector.defect_training import DefectTrainer

class UsersView(generics.CreateAPIView):
    queryset = Users.objects.all()
    serializer_class = UserSerializer

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def save_file(file, upload_dir):
    file_path = os.path.join(upload_dir, file.name)
    with default_storage.open(file_path, "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return file_path

def extract_zip(file, extract_to):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_image(img, size=(200, 200)):
    h, w = img.shape[:2]
    sub_images = []
    for y in range(0, h, size[1]):
        for x in range(0, w, size[0]):
            sub_img = img[y:y+size[1], x:x+size[0]]
            sub_images.append(sub_img)

    return sub_images

def list_files_in_directory(directory_path):
    try:
        # List all files and directories in the specified path
        items = os.listdir(directory_path)
        
        # Filter out directories, keeping only files
        files = [f for f in items if os.path.isfile(os.path.join(directory_path, f))]
        
        return files
    except Exception as e:
        return str(e)

def get_model_summary(model):
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

@csrf_exempt
def dashboard(request):
    return Response()

@csrf_exempt
def predict(request):
    if request.method == "POST":
        if 'file' not in request.FILES:
            return HttpResponseBadRequest("No file part")
        
        file = request.FILES['file']
        if(allowed_file(file.name, {'png', 'jpeg', 'jpg', 'zip'})):
            file_path = save_file(file, settings.PREDICTION_UPLOAD_DIR)

            if file.name.endswith('.zip'):
                extract_zip(file_path, settings.PREDICTION_UPLOAD_DIR)

            else:
                img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOUR)
                if img.shape[:2] != (200, 200):
                    sub_images = split_image(img)
                    result = DefectDetector.compute_prediction(sub_images)

                else:
                    result = DefectDetector.compute_prediction(file)

            return JsonResponse({'status': 'success', 'result': result})

        else:
            return HttpResponseBadRequest("File type not allowed")

@csrf_exempt
def models(request):
    if request.method == 'GET':
        models = list_files_in_directory(settings.TRAINED_MODEL_DIR)
        return JsonResponse({'status': 'success', 'models': models})
    
    return HttpResponseForbidden("Access Denied")

@csrf_exempt
def model_info(request, model_name):
    if request.method == 'GET':
        model_path = os.path.join(settings.TRAINED_MODEL_DIR, model_name)
        
        if not os.path.exists(model_path):
            return HttpResponseServerError(f"Model file not found at {model_path}")
        
        try:
            model = load_model(model_path)
            summary = get_model_summary(model)
            return JsonResponse({'status': 'success', 'model_summary': summary})
        except Exception as e:
            return HttpResponseServerError({'status': 'error', 'message': str(e)})
    pass

@csrf_exempt
def training(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return HttpResponseBadRequest("No file part")
        
        file = request.FILES['file']
        if allowed_file(file.name, {'zip'}):
            file_path = save_file(file, settings.TRAINING_UPLOAD_DIR)
            extract_zip(file_path, settings.TRAINING_UPLOAD_DIR)
            result = DefectTrainer.update_existing_model()
            return JsonResponse(result)
        
        else:
            return HttpResponseBadRequest("File type not allowed")