import os
import cv2
import numpy as np
import os.path as osp

from process_densepose import DenseposeProcesser

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
	return any(filename.endswith(extension) for extension \
			   in IMG_EXTENSIONS)

def FDP_get_IUV(image_file, file_name_pre, IUV_res_dir):

	'''
	DensePose -> IUV
	'''
	img = cv2.imread(image_file)
	H, W, _ = img.shape
	# img = img[:, :int(W/2)]

	dp_processer = DenseposeProcesser()
	iuvs, boxes = dp_processer.forward_iuv(img)
	for i in range(len(iuvs)):
		arr_iuv = np.array(iuvs[i])
		save_name = osp.join(IUV_res_dir, file_name_pre)
		np.save(save_name, arr_iuv)


if __name__ == '__main__':
	IUV_res_dir = '.test_data/dp'
	if not osp.exists(IUV_res_dir):
		os.makedirs(IUV_res_dir)

	image_path = './test_data/ori'
	list_path = os.walk(image_path)

	for path, list_dir, file_list in list_path:
		for file_name in file_list:
			image_file = osp.join(path, file_name)
			if not is_image_file(image_file):
				continue
			print('Process {}'.format(image_file))
			file_name_pre = file_name.split('.')[0]

			FDP_get_IUV(image_file, file_name_pre, IUV_res_dir)
