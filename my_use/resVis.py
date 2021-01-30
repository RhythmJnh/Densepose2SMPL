import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np

im  = cv2.imread('/home/jnh/workspace/other/dp2smpl_demo/DensePoseData/demo_data_2/im.jpg')
IUV = cv2.imread('/home/jnh/workspace/other/dp2smpl_demo/DensePoseData/demo_data_2/IUV.png')
INDS = cv2.imread('/home/jnh/workspace/other/dp2smpl_demo/DensePoseData/demo_data_2/INDS.png',  0)

fig = plt.figure(figsize=[15,15])
plt.imshow(   np.hstack((IUV[:,:,0]/24. ,IUV[:,:,1]/256. ,IUV[:,:,2]/256.))  )
plt.title('I, U and V images.')
plt.axis('off') ; plt.show()

fig = plt.figure(figsize=[12,12])
plt.imshow( im[:,:,::-1] )
plt.contour( INDS, linewidths = 4 )
plt.axis('off') ; plt.show()