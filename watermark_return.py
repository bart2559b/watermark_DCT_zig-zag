import cv2
import numpy as np
import scipy.fftpack
import os

# ฟังก์ชันสำหรับการทำ DCT 2D
def dct_2d(image):
    return scipy.fftpack.dct(scipy.fftpack.dct(image.T, norm='ortho').T, norm='ortho')

# ฟังก์ชันสำหรับการทำ inverse DCT 2D
def idct_2d(image):
    return scipy.fftpack.idct(scipy.fftpack.idct(image.T, norm='ortho').T, norm='ortho')

# ฟังก์ชันสำหรับการเบลอภาพ (Gaussian Blur)
def apply_blur(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

# ฟังก์ชันสำหรับการแบ่งภาพเป็นบล็อกและทำ DCT
def process_image_with_dct(image,size):
    h, w = image.shape
    dct_image = np.zeros_like(image)
    
    # แบ่งภาพออกเป็นบล็อก sizexsize และทำ DCT
    for i in range(0, h - h % size, size):  # แบ่งภาพออกเป็นบล็อก sizexsize
        for j in range(0, w - w % size, size):
            block = image[i:i+size, j:j+size]
            dct_block = dct_2d(block)
            dct_image[i:i+size, j:j+size] = dct_block
    
    return dct_image

# ฟังก์ชันสำหรับทำ Exclusive OR (XOR)
def xor_images(image1, image2):
    return cv2.bitwise_xor(image1, image2)

# โหลดภาพที่ฝังลายน้ำ
image_path = 'watermarked_image_color.png'
if not os.path.isfile(image_path):
    print(f"Error: The file {image_path} does not exist or the path is incorrect.")
else:
    # อ่านภาพเป็นขาวดำ
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Image could not be loaded. Please check the file path or integrity of the image.")
    else:
        s = 128
        # ประมวลผลด้วย DCT ตรงๆ
        dct_image = process_image_with_dct(image,s)

        # ประมวลผลด้วยการเบลอก่อนแล้วทำ DCT
        blurred_image = apply_blur(image)
        dct_blurred_image = process_image_with_dct(blurred_image,s)

        # ทำ Exclusive OR (XOR) ระหว่างภาพทั้งสอง
        watermark_extracted = xor_images(dct_image, dct_blurred_image)

        # ปรับค่าลายน้ำเพื่อให้แสดงผลชัดเจนขึ้น
        watermark_extracted = np.clip(watermark_extracted, 0, 255)
        watermark_extracted = np.uint8(watermark_extracted)

        # แสดงภาพลายน้ำที่ดึงออกมา
        cv2.imshow('Extracted Watermark', watermark_extracted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # บันทึกภาพลายน้ำที่ดึงออกมา
        cv2.imwrite('extracted_watermark_xor.png', watermark_extracted)
