import time,board,busio
import numpy as np
import mlx.mlx90640 as mlx
from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

dev = mlx.Mlx9064x("I2C-1", i2c_addr=0x33, frame_rate = 32.0)
dev.init()
mlx_shape = (24, 32)

x_offset = -70
y_offset = -40
scale_factor = 1.5

alpha = 0.7

while True:
	t1 = time.monotonic()
	vis = picam2.capture_array()
	vis = cv2.cvtColor(vis, cv2.COLOR_RGBA2RGB)
	try:
		# thermal capture
		raw = dev.read_frame()
		temps = dev.do_compensation(raw)
		temps_2d = np.reshape(temps, mlx_shape)
		frame_resized = cv2.resize(temps_2d, (640, 480), interpolation=cv2.INTER_LINEAR)
		frame_norm = cv2.normalize(frame_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		frame_flip = cv2.flip(frame_norm, 1)
		frame_blur = cv2.GaussianBlur(frame_flip, ksize=(5, 5), sigmaX=0)
		frame_color = cv2.applyColorMap(frame_blur, cv2.COLORMAP_INFERNO)
		
		# Scale and align 
		rows, cols, _ = frame_color.shape
		M_scale = cv2.getRotationMatrix2D((cols/2, rows/2), angle=0, scale=scale_factor)
		frame_scaled = cv2.warpAffine(frame_color, M_scale, (cols, rows))
		
		M_translate = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
		frame_aligned = cv2.warpAffine(frame_scaled, M_translate, (cols, rows))
		
		# edge capture from vis
		gray = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
		edges = cv2.Canny(gray, 80, 150)
		edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
		edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
		
		# edge mask from thermal and vis
		edges_color = cv2.bitwise_and(frame_aligned, edges_color)
		
		fusion = cv2.addWeighted(vis, 1 - alpha, frame_aligned, alpha, 0)
		fusion = cv2.addWeighted(fusion, 0.8, edges_color, 0.5, 0)
		cv2.imshow('Thermal', fusion)
		cv2.waitKey(1)
	except Exception as e:
		print(f"Frame: error: {e}")
		time.sleep(0.1)
		continue # if error, just read again

cap.release()
cv2.destroyAllWindows()
