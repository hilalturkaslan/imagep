import cv2
import numpy as np
import gradio as gr
import os

# Farklı filtre fonksiyonları
def apply_gaussian_blur(frame, ksize):
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)

def apply_sharpening_filter(frame):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def apply_edge_detection(frame):
    return cv2.Canny(frame, 100, 200)

def apply_histogram_equalization(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_frame)

def apply_binarization(frame, threshold=127):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    return binary_frame

def apply_invert_filter(frame):
    return cv2.bitwise_not(frame)

def adjust_brightness_contrast(frame, alpha=1.0, beta=50):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_grayscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_sepia_filter(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(frame, sepia_filter)

def apply_color_adjustment(frame, saturation=1.0, hue=0):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 0] = np.clip(hsv[..., 0] + hue, 0, 179)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_upscale(frame, scale_factor=2):
    height, width = frame.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_CUBIC)

def apply_multiple_filters(frame, filters, threshold, blur_size, upscale_factor):
    for filter_type in filters:
        if filter_type == "Gaussian Blur":
            frame = apply_gaussian_blur(frame, blur_size)
        elif filter_type == "Sharpen":
            frame = apply_sharpening_filter(frame)
        elif filter_type == "Edge Detection":
            frame = apply_edge_detection(frame)
        elif filter_type == "Histogram Equalization":
            frame = apply_histogram_equalization(frame)
        elif filter_type == "Binarization":
            frame = apply_binarization(frame, threshold)
        elif filter_type == "Invert":
            frame = apply_invert_filter(frame)
        elif filter_type == "Grayscale":
            frame = apply_grayscale_filter(frame)
        elif filter_type == "Sepia":
            frame = apply_sepia_filter(frame)
        elif filter_type == "Color Adjustment":
            frame = apply_color_adjustment(frame, saturation=1.5)  # Örnek doygunluk
    
    # Upscale işlemi
    frame = apply_upscale(frame, upscale_factor)
    return frame

# Filtre uygulama fonksiyonu
def apply_filter(selected_filters, input_image=None, brightness=50, contrast=1.0, threshold=127, blur_size=15, upscale_factor=2):
    if input_image is not None:
        frame = input_image
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Web kameradan görüntü alınamadı"

    # Parlaklık ve kontrast ayarlama
    frame = adjust_brightness_contrast(frame, alpha=contrast, beta=brightness)
    frame = apply_multiple_filters(frame, selected_filters, threshold, blur_size, upscale_factor)
    
    # Kaydetme işlemi
    output_path = "filtered_image.png"
    cv2.imwrite(output_path, frame)
    return frame, output_path

# Geri alma işlevi
def undo_last_action(last_frame):
    return last_frame

# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# Web Kameradan Canlı Filtreleme")

    # Filtre seçenekleri
    selected_filters = gr.CheckboxGroup(
        label="Filtre Seçin",
        choices=["Gaussian Blur", "Sharpen", "Edge Detection", "Histogram Equalization", "Binarization", "Invert", "Grayscale", "Sepia", "Color Adjustment"],
        value=["Gaussian Blur"]
    )

    # Görüntü yükleme alanı
    input_image = gr.Image(label="Resim Yükle", type="numpy")

    # Parlaklık, kontrast, eşik, blur boyutu ve upscale faktörü ayarları
    brightness_slider = gr.Slider(label="Parlaklık", minimum=0, maximum=100, value=50)
    contrast_slider = gr.Slider(label="Kontrast", minimum=0, maximum=3.0, value=1.0, step=0.1)
    threshold_slider = gr.Slider(label="Binarizasyon Eşiği", minimum=0, maximum=255, value=127)
    blur_size_slider = gr.Slider(label="Gaussian Blur Boyutu", minimum=1, maximum=30, value=15, step=2)
    upscale_factor_slider = gr.Slider(label="Upscale Faktörü", minimum=1, maximum=4, value=2, step=1)

    # Çıktı için görüntü ve dosya
    output_image = gr.Image(label="Filtre Uygulandı")
    output_file = gr.File(label="Kaydedilen Görüntü")

    # Geri alma butonu
    undo_button = gr.Button("Geri Al")

    # Filtre uygula butonu
    apply_button = gr.Button("Filtreyi Uygula")

    # Butona tıklanınca filtre uygulama fonksiyonu
    apply_button.click(fn=apply_filter, inputs=[selected_filters, input_image, brightness_slider, contrast_slider, threshold_slider, blur_size_slider, upscale_factor_slider], outputs=[output_image, output_file])

    # Geri alma butonu tıklanınca
    undo_button.click(fn=undo_last_action, inputs=output_image, outputs=output_image)

# Gradio arayüzünü başlat
demo.launch()
