import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.Image
from torchvision import transforms

def resize_img(img_path,target_res,save_name):

    img = Image.open(img_path)

    transform_bicub = transforms.Compose([
        transforms.Resize((target_res, target_res), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        ])

    img_target = transform_bicub(img)
    transforms.ToPILImage()(img_target.squeeze()).save(save_name)

img_path='./imgs_net/net1024_v8_10step_lr001_000500_001024_8.png'
# resize_img(img_path,512,'00018_res512.png')
img = cv2.imread(img_path,0)
img_float32 = np.float32(img)
cv2.imshow("img",img)
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
print(dft_shift.shape)
# 得到灰度图能表示的形式
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
