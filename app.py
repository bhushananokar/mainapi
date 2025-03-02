import cv2
import numpy as np
import base64
import io
import time
import os
import requests
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_cors import CORS

# Change this to your deployed backend URL
SERVER_URL = "https://your-backend-server.com"  # Update this

app = Flask(__name__, template_folder='.')
CORS(app)

# Ensure captures directory exists
os.makedirs("captures", exist_ok=True)

def get_skin_mask(img):
    # Rest of the code remains the same as in the original file
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    skin = cv2.bitwise_and(img, img, mask=mask)
    
    return mask, skin

def has_enough_skin(mask, min_threshold=15.0):
    total = mask.shape[0] * mask.shape[1]
    skin_count = cv2.countNonZero(mask)
    skin_percent = (skin_count / total) * 100
    
    return skin_percent >= min_threshold

def get_features(img):
    # Rest of the code remains the same as in the original file
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    detector = cv2.ORB_create(nfeatures=1000, 
                          scaleFactor=1.2,
                          nlevels=8,
                          edgeThreshold=31)
    
    kpts, descs = detector.detectAndCompute(enhanced, None)
    
    viz_img = cv2.drawKeypoints(img, kpts, None, 
                               color=(0, 255, 0), 
                               flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    
    return kpts, descs, viz_img

def find_region(img, kpts, min_points=10):
    # Original code remains the same
    if not kpts or len(kpts) < min_points:
        return None
    
    heat = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for pt in kpts:
        x, y = int(pt.pt[0]), int(pt.pt[1])
        radius = int(pt.size)
        cv2.circle(heat, (x, y), radius, 255, -1)
    
    heat = cv2.GaussianBlur(heat, (15, 15), 0)
    
    _, thresh = cv2.threshold(heat, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(biggest) < 100:
            return None
            
        x, y, w, h = cv2.boundingRect(biggest)
        
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        return (x, y, w, h)
    
    return None

def get_prediction(img_crop, domain):
    # For Vercel, this might need to be replaced with a separate service call
    # Since OpenCV and video processing might be challenging in serverless
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG')
        buf.seek(0)
        
        endpoint = f"{SERVER_URL}/{domain}/predict"
        files = {'file': ('image.jpg', buf, 'image/jpeg')}
        params = {'top_k': 1}
        
        response = requests.post(endpoint, files=files, params=params, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            preds = result.get('predictions', [])
            
            if preds:
                return preds[0]['class'], preds[0]['probability']
        
        return "Server Error", 0.0
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0

class VideoProcessor:
    _instance = None

    def __new__(cls):
        # Note: This might not work exactly the same in a serverless environment
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.last_check = time.time()
            cls._instance.curr_result = ("No skin detected", 0.0)
            cls._instance.current_domain = 'skin'
        return cls._instance
    
    def process_frame(self):
        # This might need to be adjusted for serverless
        # Consider using a mock or alternative method for Vercel
        return None, None, None
    
    def change_domain(self, domain):
        self.current_domain = domain

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/video_feed')
def video_feed():
    # Note: Video streaming might not work well in serverless
    # Consider alternative visualization methods
    return "Video streaming not supported in this deployment"

@app.route('/change_domain', methods=['POST'])
def change_domain():
    domain = request.form.get('domain', 'skin')
    VideoProcessor().change_domain(domain)
    return jsonify({"status": "success", "domain": domain})

# For Vercel serverless function
def handler(event, context):
    # Placeholder for serverless handler if needed
    return {"statusCode": 200, "body": "Medical Image Classifier"}

# This allows running the app locally for testing
if __name__ == '__main__':
    app.run(debug=True)
