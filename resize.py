import cv2

img = cv2.imread('/media/test/run/count/countx/CounTX-main-arg-4-conv-loss-auxloss/121212.jpg')
img=cv2.resize(img,(384,384))
cv2.imwrite('/media/test/run/count/countx/CounTX-main-arg-4-conv-loss-auxloss/121212.jpg',img)
