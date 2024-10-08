import cv2
import numpy as np
import scipy.fftpack
import os  # นำเข้า os module

# ฟังก์ชันสำหรับการทำ zig-zag scan (ขนาด 8x8)
def zigzag_indices(n):
    indices = np.arange(n*n).reshape(n, n)
    idx_list = []
    
    for i in range(2*n - 1):
        if i % 2 == 0:
            idx_list.extend(np.diag(np.fliplr(indices), i-n+1).tolist())
        else:
            idx_list.extend(np.diag(indices, i-n+1).tolist())
            
    return idx_list

# ฟังก์ชันฝังลายน้ำลงใน DCT ของภาพแต่ละช่องสี (ขนาด 8x8)
def embed_watermark(image_channel, watermark_channel,size, alpha=100):  # ลดค่า alpha เพื่อเพิ่มความเนียน
    h, w = image_channel.shape
    watermarked_channel = np.zeros_like(image_channel)
    
    # แปลงภาพเป็นบล็อก 8x8 และทำ DCT
    for i in range(0, h, size):
        for j in range(0, w, size):
            block = image_channel[i:i+size, j:j+size]
            
            # ตรวจสอบว่าขนาดบล็อกเป็น 8x8 หรือไม่
            if block.shape[0] != size or block.shape[1] != size:
                continue  # ข้ามบล็อกที่ไม่ใช่ขนาด 8x8
            
            dct_block = scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            
            # Zig-zag scan
            zigzag = zigzag_indices(size)
            
            # ฝังลายน้ำลงในค่าสัมประสิทธิ์ DCT
            for k in range(min(len(zigzag), len(watermark_channel))):
                idx = zigzag[k]
                x, y = idx // size, idx % size  # ปรับให้เข้ากับบล็อกขนาด 8x8
                dct_block[x, y] += alpha * watermark_channel[k]
            
            # inverse DCT
            idct_block = scipy.fftpack.idct(scipy.fftpack.idct(dct_block.T, norm='ortho').T, norm='ortho')
            
            # กรองภาพหลังจาก IDCT
            idct_block = np.clip(idct_block, 0, 255)  # จำกัดค่าให้ภายในช่วง [0, 255]
            
            watermarked_channel[i:i+size, j:j+size] = idct_block
    
    return watermarked_channel

# ฟังก์ชันเพื่อโหลดภาพที่รองรับทั้ง PNG และ JPG
def load_image(filename, color_mode=cv2.IMREAD_COLOR):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"ไม่พบไฟล์ {filename}")
    image = cv2.imread(filename, color_mode)
    if image is None:
        raise ValueError(f"ไม่สามารถเปิดไฟล์ {filename} ได้ กรุณาตรวจสอบเส้นทางไฟล์")
    return image

# โหลดภาพต้นฉบับเป็นภาพสี
image = load_image('image.png')  # เปลี่ยนเป็น .jpg หรือ .png ตามต้องการ
# โหลดลายน้ำ
watermark = load_image('watermark2.png', cv2.IMREAD_GRAYSCALE)  # เปลี่ยนเป็น .jpg หรือ .png ตามต้องการ

s = 128

# แยกภาพออกเป็นช่องสี (BGR)
b_channel, g_channel, r_channel = cv2.split(image)

# ปรับขนาดลายน้ำให้ตรงกับขนาดของภาพต้นฉบับ
watermark_resized = cv2.resize(watermark, (image.shape[1], image.shape[0]))

# ฝังลายน้ำในแต่ละช่องสี
# watermarked_b = b_channel
# watermarked_g = g_channel
# watermarked_r = r_channel
watermarked_b = embed_watermark(b_channel, watermark_resized.flatten(),s)
watermarked_g = embed_watermark(g_channel, watermark_resized.flatten(),s)
watermarked_r = embed_watermark(r_channel, watermark_resized.flatten(),s)

# รวมภาพช่องสีที่ฝังลายน้ำแล้วกลับมาเป็นภาพสี
watermarked_image = cv2.merge((watermarked_b, watermarked_g, watermarked_r))

# บันทึกผลลัพธ์
cv2.imwrite('watermarked_image_color.png', watermarked_image)  # เปลี่ยนเป็น .jpg หรือ .png ตามต้องการ
cv2.imwrite('watermarked_b.png', watermarked_b)
cv2.imwrite('watermarked_g.png', watermarked_g)
cv2.imwrite('watermarked_r.png', watermarked_r)