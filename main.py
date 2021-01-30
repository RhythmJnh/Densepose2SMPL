#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import time
import os.path as osp
# import smplx
import cv2
from tqdm import tqdm
from collections import defaultdict
import PIL.Image as pil_img

from my_use import densepose_methods as dp_utils
from my_use.camera import create_camera
from my_use import fitting
from my_use.prior import create_prior
# from human_body_prior.tools.model_loader import load_vposer
from optimizers import optim_factory
from sklearn.cluster import KMeans	
from my_use.utils import render
from model.daz import DazModel

import torch
torch.backends.cudnn.enabled = False

focal        = 5000 * 28. / 234.7 * 2
iter_stage   = 2
batch_size   = 1
dtype        = torch.float32
# gender       = 'neutral'
gender       = 'female'
# gender       = 'male'
visualize    = True

data_folder  = './test_data'
res_folder   = './test_res'

interactive      = True

show_loss        = False
show_loss_type   = 'pro_loss'

sample_points    = 10
sample_contours  = 100
device = torch.device(f"cuda:{0}")


def iuv_fitting(body_model, camera, \
	img_ori, fit_map_xy, face_index, \
	bary_coords, edge_idxs, map_xy_cam, face_cam, \
	bary_cam, img_res, param_res, check_body_back, \
	):

	# calculate init Z
	init_t = fitting.guess_init(body_model, 
								edge_idxs, 
								dtype=dtype,
								device=device,
								focal_length=focal,
								map_xy_now=map_xy_cam, 
								face_now=face_cam, 
								bary_now=bary_cam, 
								)

	camera_loss = fitting.create_loss('camera_init',
									  trans_estimation=init_t,
									  depth_loss_weight=1e2,
									  dtype=dtype,
									  ).to(device=device)

	camera_loss.trans_estimation[:] = init_t	


	shape_prior = create_prior(prior_type='l2', dtype=dtype)

	loss = fitting.create_loss(loss_type='smplify',
							   rho=100,
							   shape_prior=shape_prior,
							   dtype=dtype,
							   )

	loss = loss.to(device=device)

	cam_need_xy = torch.unsqueeze(torch.tensor(map_xy_cam, device=device, dtype=dtype), 0)

	gt_faces = face_index
	gt_coords = bary_coords
	gt_xy = torch.tensor(fit_map_xy, device=device, dtype=dtype)

	# Weights used for the pose prior and the shape prior
	data_weights = [1.] * 5
	body_pose_prior_weights = [404, 404, 57.4, 4.78, 4.78]
	shape_weights = [100, 50, 10, 5, 5]

	opt_weights_dict = {'data_weight': data_weights,
						'body_pose_weight': body_pose_prior_weights,
						'shape_weight': shape_weights,
						}

	keys = opt_weights_dict.keys()
	opt_weights = [dict(zip(keys, vals)) for vals in
				   zip(*(opt_weights_dict[k] for k in keys
						 if opt_weights_dict[k] is not None))]
	for weight_list in opt_weights:
		for key in weight_list:
			weight_list[key] = torch.tensor(weight_list[key],
											device=device,
											dtype=dtype)

	with fitting.FittingMonitor(visualize=visualize) as monitor:

		img = torch.tensor(img_ori, dtype=dtype)
		H, W, _ = img.shape

		data_weight = 1000 / H
		# Reset the parameters to estimate the initial translation of the
		# The closure passed to the optimizer
		camera_loss.reset_loss_weights({'data_weight': data_weight})

		# Update the value of the translation of the camera as well as
		# the image center.
		with torch.no_grad():
			camera.translation[:] = init_t.view_as(camera.translation)
			camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

		# Re-enable gradient calculation for the camera translation
		camera.translation.requires_grad = True

		init_go_pose = torch.zeros([1, 3]).to(device)
		init_go_pose.requires_grad = True
		camera_opt_params = [camera.translation, init_go_pose]

		camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
			camera_opt_params)

		# The closure passed to the optimizer
		fit_camera = monitor.create_fitting_closure(
				camera_optimizer, body_model, camera, cam_need_xy,
				face_cam, bary_cam,
				camera_loss, create_graph=camera_create_graph, go_pose=init_go_pose)
		# Step 1: Optimize over the torso joints the camera translation
		# Initialize the computational graph by feeding the initial translation
		# of the camera and the initial pose of the body model.
		camera_init_start = time.time()
		if interactive:
			print("Optimize camera params...")
		cam_init_loss_val = monitor.run_fitting(camera_optimizer,
												fit_camera,
												body_model, go_pose=init_go_pose)

		if interactive:
			if torch.cuda.is_available():
				torch.cuda.synchronize()
			tqdm.write('Camera initialization done after {:.4f}'.format(
					time.time() - camera_init_start))
			tqdm.write('Camera initialization final loss {:.4f}'.format(
					cam_init_loss_val))

		if check_body_back:
			body_orient = init_go_pose.detach().cpu().numpy()
			flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
				cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
			flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

			flipped_orient = torch.tensor(flipped_orient,
										  dtype=dtype,
										  device=device).unsqueeze(dim=0)
			orientations = [body_orient, flipped_orient]
		else:
			orientations = [init_go_pose.detach().cpu().numpy()]

		# store here the final error for both orientations,
		# and pick the orientation resulting in the lowest error
		results = []

		# Step 2: Optimize the full model
		if interactive:
			print("Optimize full model...")
		final_loss_val = 0
		body_poses = []
		for or_idx, orient in enumerate(orientations):
			opt_start = time.time()
			opt_body_pose = torch.zeros([1, 169, 3]).to(device)
			opt_go_pose = torch.from_numpy(orient.astype(np.float32)).to(device)
			opt_betas = torch.zeros([1, 21]).to(device)
			opt_body_pose.requires_grad=True
			opt_go_pose.requires_grad=True
			# opt_betas.requires_grad=True

			# Loss curve
			plt=None
			if show_loss:
				import matplotlib.pyplot as plt
				plt.ion()
				plt.figure()
				plt.title('{}'.format(show_loss_type))

			for opt_idx, curr_weights in enumerate(opt_weights):


				body_optimizer, body_create_graph = optim_factory.create_optimizer([opt_betas,
																					opt_go_pose,
																					opt_body_pose,
																					camera.translation])
				body_optimizer.zero_grad()

				curr_weights['data_weight'] = data_weight

				loss.reset_loss_weights(curr_weights)

				closure = monitor.create_fitting_closure(
					body_optimizer, body_model,
					camera=camera, gt_xy=gt_xy,
					xy2face_inds=gt_faces,
					xy2bary_coords=gt_coords,
					plt=plt,
					show_loss=show_loss,
					show_loss_type=show_loss_type,
					loss=loss, create_graph=body_create_graph,
					body_pose=opt_body_pose,
					go_pose=opt_go_pose,
					betas=opt_betas,
					)

				if interactive:
					if torch.cuda.is_available():
						torch.cuda.synchronize()
					stage_start = time.time()

				final_loss_val = monitor.run_fitting(
					body_optimizer,
					closure,
					body_model,
				    opt_body_pose,
					opt_go_pose,
					opt_betas,
				)

				if interactive:
					if torch.cuda.is_available():
						torch.cuda.synchronize()
					elapsed = time.time() - stage_start
					if interactive:
						tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
							opt_idx, elapsed))
			if show_loss:
				plt.close()

			if interactive:
				if torch.cuda.is_available():
					torch.cuda.synchronize()
				elapsed = time.time() - opt_start
				tqdm.write(
					'Body fitting Orientation {} done after {:.4f} seconds'.format(
						or_idx, elapsed))
				tqdm.write('Body final loss val = {:.5f}'.format(
					final_loss_val))


		# with open(param_res, 'wb') as result_file:
		# 	if len(results) > 1:
		# 		min_idx = (0 if results[0]['loss'] < results[1]['loss']
		# 				   else 1)
		# 	else:
		# 		min_idx = 0
		# 	pk.dump(results[min_idx]['result'], result_file, protocol=2)

	vers, _, _ = body_model(global_orient=opt_go_pose,
							body_pose=opt_body_pose,
							betas=opt_betas)
	vertices = vers.detach().cpu().numpy().squeeze()

	color = render(vertices, body_model.faces, camera, 
				   focal=focal, depth_or_color='color',H=H, W=W)
	color = color.astype(np.float32) / 255.0

	valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
	input_img = img.detach().cpu().numpy()
	output_img = (color[:, :, :-1] * valid_mask +
				 (1 - valid_mask) * input_img)

	img = pil_img.fromarray((output_img * 255).astype(np.uint8))
	img.save(img_res)

	return 0


def get_contour_pt_face_bary(iuv_png, map_iuv, map_xy):

	# get contours
	img = cv2.cvtColor(iuv_png, cv2.COLOR_BGR2GRAY)
	ret, binary = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours = np.array(contours)

	contours_xy = []
	for i in range(contours.shape[0]):
		contours_xy.append(contours[i].reshape([-1,2]))
	contours_xy = np.vstack(contours_xy)

	# sampl point in contours
	sample_ind = np.arange(0, contours_xy.shape[0], \
		int(contours_xy.shape[0]/sample_contours))

	contours_xy = contours_xy[sample_ind]

	con_map_ind = []
	for i in range(contours_xy.shape[0]):
		now_xy = contours_xy[i]
		try: 
			now_id = np.argwhere((map_xy[:, 0] == now_xy[0]) & \
				(map_xy[:, 1] == now_xy[1]))[0][0]
		except:
			continue
		else:
			con_map_ind.append(now_id)

	con_map_ind = np.array(con_map_ind).reshape([1, -1])

	con_map_uv = map_iuv[con_map_ind]
	con_map_xy = map_xy[con_map_ind]

	face_con , bary_con = calculaterFB(con_map_uv[0])

	return face_con, bary_con, con_map_xy


def get_cam_need(map_iuv, map_xy, iuv_png):
	# select chest part
	chest_part = [3, 8]
	chest_fron = np.where(map_iuv[:,0] == 3)[0]
	chest_back = np.where(map_iuv[:,0] == 8)[0]

	check_body_back = len(chest_fron)<len(chest_back)

	sel_cam_part = chest_part[0] if not check_body_back \
		else chest_part[1]

	sel_cam_total = map_xy[chest_fron] if sel_cam_part == chest_part[0] \
		else map_xy[chest_back]

	# get chest box
	box_h_count = np.bincount(sel_cam_total[:, 0])
	box_h_count_max = np.argmax(box_h_count)
	box_h_index = np.where(sel_cam_total[:, 0] == box_h_count_max)[0]
	box_A = sel_cam_total[box_h_index[0]]
	box_B = sel_cam_total[box_h_index[-1]]

	ind_A = np.argwhere((map_xy[:, 0] == box_A[0]) & (map_xy[:, 1] == box_A[1]))[0][0]
	ind_B = np.argwhere((map_xy[:, 0] == box_B[0]) & (map_xy[:, 1] == box_B[1]))[0][0]

	box_w_count = np.bincount(sel_cam_total[:, 1])
	box_w_count_max = np.argmax(box_w_count)
	box_w_index = np.where(sel_cam_total[:, 1] == box_w_count_max)[0]
	box_C = sel_cam_total[box_w_index[0]]
	box_D = sel_cam_total[box_w_index[-1]]

	ind_C = np.argwhere((map_xy[:, 0] == box_C[0]) & (map_xy[:, 1] == box_C[1]))[0][0]
	ind_D = np.argwhere((map_xy[:, 0] == box_D[0]) & (map_xy[:, 1] == box_D[1]))[0][0]

	edge_index = [ind_A, ind_B, ind_C, ind_D]	
	edge_idxs = [[0, 1], [2, 3]]

	map_xy_cam = map_xy[edge_index]
	map_uv_cam = map_iuv[edge_index]

	# for i in range(1, 4):
	# 	cv2.circle(iuv_png, (map_xy_cam[i][0], map_xy_cam[i][1]), radius=5, color=[255, 0, 0], thickness=3)
	# 	cv2.imshow('test', iuv_png)
	# 	cv2.waitKey()
	# exit()

	face_cam , bary_cam = calculaterFB(map_uv_cam)

	return edge_idxs, face_cam, bary_cam, map_xy_cam, check_body_back


def cvShow(img):
	cv2.namedWindow('show', 0)
	# cv2.resizeWindow('show', 640, 480)
	cv2.imshow('show', img)
	cv2.waitKey(1)
	# cv2.destroyAllWindows()


def drawPts2img(pts, img):
	import copy
	draw_img = copy.deepcopy(img)
	for i in range(pts.shape[0]):
		cv2.circle(draw_img, (int(pts[i, 0]), int(pts[i, 1])), 2, (255, 0, 0, 255), -1)
		# cv2.putText(draw_img, str(i), (int(pts[i, 0] + 2), int(pts[i, 1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
	cvShow(draw_img)


def get_map_iuv_xy(iuv_png):

	xy_ind = np.where(iuv_png[:,:,0]!=0)
	map_iuv = iuv_png[xy_ind]
	map_xy = np.array(xy_ind).T
	map_xy = map_xy[::, [1, 0]]

	return map_iuv, map_xy


def sample_fit_pt_face_bary(map_iuv, map_xy):

	part_map = []
	# generate fit usage data
	for i in range(1, 25):
		part_index = np.where(map_iuv[:,0] == i)[0]
		part_len = len(part_index)
		if part_len < sample_points*88:
			part_map.append(None)
			continue

		part_map_xy = map_xy[part_index]
		if sample_points>part_len:
			kmeans = KMeans(n_clusters=part_len, random_state=0).fit(part_map_xy)
		else:
			kmeans = KMeans(n_clusters=sample_points, random_state=0).fit(part_map_xy)
		part_map_xy = kmeans.cluster_centers_.astype(int)
		part_map.append(part_map_xy)

	part_map = np.array(part_map)

	part_map_index = []
	for i in range(24):
		if part_map[i] is None:
			continue
		for j in range(len(part_map[i])):
			now_xy = part_map[i][j]
			try:
				now_id = np.argwhere((map_xy[:, 0] == now_xy[0]) & \
					(map_xy[:, 1] == now_xy[1]))[0][0]
			except:
				continue
			else:
				part_map_index.append(now_id)
	part_map_index = np.array(part_map_index).reshape([1, -1])

	fit_map_xy = map_xy[part_map_index]
	fit_map_uv = map_iuv[part_map_index]

	face_index , bary_coords = calculaterFB(fit_map_uv[0])

	return face_index, bary_coords, fit_map_xy


def calculaterFB(map_iuv):
	# using densepose methods
	DP = dp_utils.DensePoseMethods()

	faceIndex = []
	baryCoords = []
	for i, (ii,uu,vv) in enumerate(map_iuv):
		# Convert IUV to FBC (faceIndex and barycentric coordinates.)
		# time1 = time.time()
		# FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii, uu, vv)
		# time2 = time.time()
		# print(time2-time1)
		# time1 = time.time()
		FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC_fast(ii, uu/255., vv/255.)
		bc = [bc1, bc2, bc3]
		# time2 = time.time()
		# print(time2-time1)
		# exit()
		# Use FBC to get 3D coordinates on the surface.
		face_index = DP.FBC2Index(FaceIndex)	
		faceIndex.append(face_index)
		baryCoords.append(bc)

	return np.array(faceIndex), np.array(baryCoords)


def main(app_start, img_ori, img_dp):

	# model initial
	# model_params = dict(model_path=model_folder,
	# 					create_body_pose=False,
	# 					create_transl=False,
	# 					dtype=dtype,
	# 					)
	#
	# smpl = smplx.create(model_type='smpl',
	# 					gender=gender, **model_params)
	daz = DazModel()


	# load camera
	camera = create_camera(focal_length_x=focal, 
						   focal_length_y=focal,
						   dtype=dtype)

	# load vposer
	# vposer, _ = load_vposer(vposer_ckpt,
	# 						vp_model='snapshot')

	# to device
	camera = camera.to(device=device)
	daz   = daz.to(device=device)
	# vposer = vposer.to(device=device)
	# vposer.eval()

	# results folder
	img_fn = osp.join(res_folder, 'images')
	param_fn = osp.join(res_folder, 'params')

	if not osp.exists(img_fn):
		os.makedirs(img_fn)

	if not osp.exists(param_fn):
		os.makedirs(param_fn)

	# load iuv map and origin image
	img_ori = cv2.imread(osp.join(data_folder+'/ori',\
	 img_ori))
	img_ori = img_ori[...,::-1]/255.0
	iuv_png = cv2.imread(osp.join(data_folder+'/dp', img_dp))
	img_res   = osp.join(img_fn, img_dp.replace('.png', '.png'))
	param_res = osp.join(param_fn, img_dp.replace('.png', '.pkl'))

	map_iuv, map_xy = get_map_iuv_xy(iuv_png)

	# test = iuv_png[:, :, 0]/24.
	# cv2.imshow('test', test)
	# cv2.waitKey()

	# get estimated camera para points
	edge_idxs, face_cam, bary_cam, map_xy_cam, check_body_back = \
		get_cam_need(map_iuv, map_xy, iuv_png)

	# sample area and get face index, barycentric
	face_index, bary_coords, fit_map_xy = sample_fit_pt_face_bary(\
		map_iuv, map_xy)

	# drawPts2img(fit_map_xy[0], iuv_png)

	# res = {}
	# res['face'] = face_index
	# res['bary'] = bary_coords
	# res['uvxy'] = fit_map_xy
	# import pickle
	# pickle.dump(res, open('check_sample_uv.pkl', 'wb'))
	# exit()

	iuv_fitting(body_model=daz,
				camera=camera,
				img_ori=img_ori,
				fit_map_xy=fit_map_xy,
				face_index=face_index,
				bary_coords=bary_coords,
				edge_idxs=edge_idxs,
				map_xy_cam=map_xy_cam,
				face_cam=face_cam,
				bary_cam=bary_cam,
				img_res=img_res, 
				param_res=param_res,
				check_body_back=check_body_back,
				)

	if interactive:
		tot_time = time.time()
		print('Total opt time: {}'.format(tot_time-app_start))


if __name__ == '__main__':
	app_start = time.time()
	print(torch.cuda.is_available())

	img_ori_p = data_folder+'/ori'
	img_dp_p = data_folder+'/dp'
	img_ori_files = sorted(os.listdir(img_ori_p))
	img_dp_files = sorted(os.listdir(img_dp_p))
	for i in range(len(img_ori_files)):
		img_ori = img_ori_files[i]
		print(img_ori)
		img_dp  = img_dp_files[i]
		main(app_start, img_ori, img_dp)