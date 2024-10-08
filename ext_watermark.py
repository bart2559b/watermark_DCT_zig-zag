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

def extract_watermark(original_channel, watermarked_channel, size, alpha=1):
    h, w = original_channel.shape
    extracted_watermark = np.zeros((h, w), dtype=np.float32)
    
    # Loop through each 8x8 block in the image
    for i in range(0, h, size):
        for j in range(0, w, size):
            original_block = original_channel[i:i+size, j:j+size]
            watermarked_block = watermarked_channel[i:i+size, j:j+size]
            
            # Make sure the block is of the correct size
            if original_block.shape[0] != size or original_block.shape[1] != size:
                continue  # Skip blocks that are not 8x8
            
            # Apply DCT to both original and watermarked blocks
            dct_original_block = scipy.fftpack.dct(scipy.fftpack.dct(original_block.T, norm='ortho').T, norm='ortho')
            dct_watermarked_block = scipy.fftpack.dct(scipy.fftpack.dct(watermarked_block.T, norm='ortho').T, norm='ortho')
            
            # Get zigzag pattern indices
            zigzag = zigzag_indices(size)
            
            # Extract watermark coefficients from the differences between DCT coefficients
            for k in range(len(zigzag)):
                idx = zigzag[k]
                x, y = idx // size, idx % size
                
                # Compute the difference between the original and watermarked DCT coefficients
                watermark_value = (dct_watermarked_block[x, y] - dct_original_block[x, y]) / alpha
                extracted_watermark[i+x, j+y] = watermark_value
    
    # Normalize the extracted watermark to be in the range [0, 255]
    extracted_watermark = np.clip(extracted_watermark, 0, 255)
    return extracted_watermark



def load_image(filename, color_mode=cv2.IMREAD_COLOR):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"ไม่พบไฟล์ {filename}")
    image = cv2.imread(filename, color_mode)
    if image is None:
        raise ValueError(f"ไม่สามารถเปิดไฟล์ {filename} ได้ กรุณาตรวจสอบเส้นทางไฟล์")
    return image

# Load original image and watermarked image
original_image = load_image('image.png')  # Original image
watermarked_image = load_image('watermarked_image_color.png')  # Watermarked image

# Split both original and watermarked images into BGR channels
orig_b, orig_g, orig_r = cv2.split(original_image)
wm_b, wm_g, wm_r = cv2.split(watermarked_image)

s = 128

# Extract watermark from each color channel
extracted_watermark_b = extract_watermark(orig_b, wm_b, s)
extracted_watermark_g = extract_watermark(orig_g, wm_g, s)
extracted_watermark_r = extract_watermark(orig_r, wm_r, s)

# Combine the extracted watermarks from all channels (you can average them or use any channel)
extracted_watermark = (extracted_watermark_b + extracted_watermark_g + extracted_watermark_r) / 3

# Save the extracted watermark
cv2.imwrite('extracted_watermark.png', extracted_watermark)
