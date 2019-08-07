from mmdet.apis import init_detector, inference_detector, show_result
import time
import os

config_file = 'configs/htc/htc_x101_64x4d_fpn_20e_16gpu.py'
checkpoint_file = 'checkpoints/htc_x101_64x4d_fpn_20e_20190408-497f2561.pth'
folder_name = config_file.split('/')[2].split('.')[0]
print('FOLDER NAME ',folder_name)
if not os.path.exists(os.path.join(os.getcwd(), folder_name)):
  os.mkdir(os.path.join(os.getcwd(), folder_name))
# build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
import cv2 
import numpy as np 
   
# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('video.mp4') 
   
# Check if camera opened successfully pr
if (cap.isOpened()== False):  
  print("Error opening video  file") 
   
# Read until video is completed 
count =0
start = time.time()
while(cap.isOpened()): 
      
  # Capture frame-by-frame
  ret, frame = cap.read() 
  if ret == True: 
    # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, frame)
    show_result(frame, result, model.CLASSES, out_file='benchmarks/{}/result{}.jpg'.format(folder_name ,count))
    count+=1
    print('Count ',count, 'Time ',time.time() - start)
end = time.time() - start
cap.release()
cv2.destroyAllWindows()
print('_______', end)
    # test a list of images and write the results to image files
    # imgs = ['test1.jpg', 'test2.jpg']
    # for i, result in enumerate(inference_detector(model, imgs)):
        # show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))