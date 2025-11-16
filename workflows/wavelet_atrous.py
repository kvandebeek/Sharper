import cv2, numpy as np

def atrous_wavelet(img, gains):
    img = img.astype(np.float32)
    cur = img
    levels = []
    for i in range(5):
        k = 3 + 4*i
        if k%2==0: k+=1
        blur = cv2.GaussianBlur(cur, (k,k), 0)
        detail = cur - blur
        levels.append(detail)
        cur = blur
    out = img.copy()
    for g, d in zip(gains, levels):
        out += d * g
    return np.clip(out,0,1)
