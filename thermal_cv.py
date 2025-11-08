import time,board,busio
import numpy as np
import mlx.mlx90640 as mlx
import cv2

dev = mlx.Mlx9064x("I2C-1", i2c_addr=0x33, frame_rate = 64.0)
dev.init()
mlx_shape = (24, 32)

while True:
    t1 = time.monotonic()
    try:
        raw = dev.read_frame()
        temps = dev.do_compensation(raw)
        temps_2d = np.reshape(temps, mlx_shape)
        frame_resized = cv2.resize(temps_2d, (320, 240), interpolation=cv2.INTER_CUBIC)
        frame_norm = cv2.normalize(frame_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #frame_flip = cv2.flip(frame_norm, 1)
        frame_blur = cv2.GaussianBlur(frame_norm, ksize=(5, 5), sigmaX=0)
        frame_color = cv2.applyColorMap(frame_norm, cv2.COLORMAP_INFERNO)
        cv2.imshow('Thermal', frame_color)
        cv2.waitKey(1)
    except Exception as e:
        print(f"Frame: error: {e}")
        time.sleep(0.1)
        continue # if error, just read again

