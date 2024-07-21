from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO
import base64
from skimage.metrics import structural_similarity
import os

app = Flask(__name__)

def compare_images(img1, img2):
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))
    
    # Renk farklılıklarını da dikkate al
    diff = cv2.absdiff(img1, img2)
    
    # Gri tonlamaya dönüştür
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Eşik değerini düşür
    threshold = 20
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Gürültüyü azalt
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Farklılıkları işaretle
    diff_mask = thresh > 0
    img1[diff_mask] = [0, 0, 255]  # Kırmızı ile işaretle
    
    # Konturları çiz
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img1, contours, -1, (0, 255, 0), 1)  # Yeşil konturlar, ince çizgi
    
    _, buffer = cv2.imencode('.jpg', img1)
    result_image = base64.b64encode(buffer).decode('utf-8')
    
    diff_pixel_count = np.sum(diff_mask)
    result_text = f"Resimler {'aynı' if diff_pixel_count == 0 else 'farklı'}. Farklı piksel sayısı: {diff_pixel_count}"
    
    return result_image, result_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'img1' not in request.files or 'img2' not in request.files:
            return "Hata: Dosyalar yüklenmedi", 400
        
        file1 = request.files['img1']
        file2 = request.files['img2']
        
        if file1.filename == '' or file2.filename == '':
            return "Hata: Dosya seçilmedi", 400
        
        # Dosyaları oku ve OpenCV formatına dönüştür
        img1_bytes = file1.read()
        img2_bytes = file2.read()
        img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        result_image, result_text = compare_images(img1, img2)
        return render_template('result.html', result_text=result_text, result_image=result_image)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))