import cv2
import numpy as np
from PIL import Image

img_file = "000001_10.png"
data = cv2.imread(img_file)

info = np.iinfo(data.dtype)  # Get the information of the incoming image type
data = data.astype(np.float64) / info.max  # normalize the data to 0 - 1
data = 255 * data  # Now scale by 255
img = data.astype(np.uint8)
cv2.imshow("Window", img)

i = Image.fromarray(np.uint8(img))
i.show()
i.save("./000001_10_n.jpg")
