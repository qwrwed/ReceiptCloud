from DewarpNet.unwarp import unwarp
import cv2
import sys

img_path = sys.argv[1]
img = cv2.imread(img_path)
img_path_split = img_path.split('.')
img_path_name = '.'.join(img_path_split[:-1])
img_path_ext = img_path_split[-1]

cv2.imshow("before", img)
img = unwarp(img, 'models/unetnc_doc3d.pkl', 'models/dnetccnl_doc3d.pkl')
cv2.imshow("after", img)
# https://stackoverflow.com/a/58404775
img = cv2.convertScaleAbs(img, alpha=(255.0)) 
cv2.imwrite(img_path_name+"-uw."+img_path_ext, img)
cv2.waitKey()