from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

def compare_images(img1, img2):
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))
    
    # Consider color differences
    diff = cv2.absdiff(img1, img2)
    
    # Convert to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Lower the threshold
    threshold = 20
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Reduce noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Mark differences
    diff_mask = thresh > 0
    img1[diff_mask] = [0, 0, 255]  # Mark with red
    
    # Draw contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img1, contours, -1, (0, 255, 0), 1)  # Green contours, thin line
    
    _, buffer = cv2.imencode('.jpg', img1)
    result_image = base64.b64encode(buffer).decode('utf-8')
    
    diff_pixel_count = np.sum(diff_mask)
    result_text = f"Images are {'the same' if diff_pixel_count == 0 else 'different'}. Different pixel count: {diff_pixel_count}"
    
    return result_image, result_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'img1' not in request.files or 'img2' not in request.files:
            return "Error: Files not uploaded", 400
        
        file1 = request.files['img1']
        file2 = request.files['img2']
        
        if file1.filename == '' or file2.filename == '':
            return "Error: No file selected", 400
        
        # Read files and convert to OpenCV format
        img1_bytes = file1.read()
        img2_bytes = file2.read()
        img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        result_image, result_text = compare_images(img1, img2)
        return render_template('result.html', result_text=result_text, result_image=result_image)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))