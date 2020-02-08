#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import time
import os.path as osp
import smplx
import cv2
from tqdm import tqdm
from collections import defaultdict
import PIL.Image as pil_img

from my_use import densepose_methods as dp_utils
from my_use.camera import create_camera
from my_use import fitting
from my_use.prior import create_prior
from human_body_prior.tools.model_loader import load_vposer
from optimizers import optim_factory
from sklearn.cluster import KMeans	
from my_use.utils import render

import torch
torch.backends.cudnn.enabled = False

focal        = 5000.
iter_stage   = 5
batch_size   = 1
dtype        = torch.float32
# gender       = 'neutral'
gender       = 'female'
# gender       = 'male'
visualize    = False

data_folder  = './test_data/'
res_folder   = './test_res'

vposer_ckpt  = './vposer_v1_0'
model_folder = './models'

part_segm_path   = ''
interpenetration = False
interactive      = True

show_loss        = False
show_loss_type   = 'pro_loss'

sample_points    = 5
sample_contours  = 100
device = torch.device('cpu')


def iuv_fitting(body_model, camera, vposer, \
	img_ori, fit_map_xy, face_index, \
	bary_coords, edge_idxs, map_xy_cam, face_cam, \
	bary_cam, img_res, param_res, check_body_back, \
	con_map_xy, face_con, bary_con,\
	):

	# mean pose
	pose_embedding = torch.zeros([batch_size, 32], 
								 dtype=dtype, 
								 device=device, 
								 requires_grad=True
								 )

	# calculate init Z
	init_t = fitting.guess_init(body_model, 
								edge_idxs, 
								dtype=dtype,
								device=device,
								pose_embedding=pose_embedding, 
								vposer=vposer, 
								model_type='smpl', 
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

	# Create the search tree
	ign_part_pairs = ["9,16", "9,17", "6,16", "6,17", "1,2", "12,22"]
	max_collisions = 128
	df_cone_heigh = 0.0001
	search_tree = None
	pen_distance = None
	filter_faces = None
	if interpenetration:
		from mesh_intersection.bvh_search_tree import BVH
		import mesh_intersection.loss as collisions_loss
		from mesh_intersection.filter_faces import FilterFaces

		assert True, 'Interpenetration term can only be used with CUDA'
		assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

		search_tree = BVH(max_collisions=max_collisions)

		pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_heigh, point2plane=False,
                vectorized=True, penalize_outside=True)

	if part_segm_path and interpenetration:
        # Read the part segmentation
		part_segm_fn = os.path.expandvars(part_segm_path)
		with open(part_segm_fn, 'rb') as faces_parents_file:
			face_segm_data = pk.load(faces_parents_file,
                                             encoding='latin1')
		faces_segm = face_segm_data['segm']
		faces_parents = face_segm_data['parents']
        # Create the module used to filter invalid collision pairs
		filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)


	angle_prior = create_prior(prior_type='angle', dtype=dtype)
	shape_prior = create_prior(prior_type='l2', dtype=dtype)

	loss = fitting.create_loss(loss_type='smplify',
                               rho=100,
                 			   angle_prior=angle_prior,
            				   shape_prior=shape_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               )

	loss = loss.to(device=device)

	body_mean_pose = torch.zeros([batch_size, 32], dtype=dtype,)

	cam_need_xy = torch.unsqueeze(torch.tensor(map_xy_cam, device=device, dtype=dtype), 0)

	gt_faces = face_index
	gt_coords = bary_coords
	gt_xy = torch.tensor(fit_map_xy, device=device, dtype=dtype)

	# contours
	con_faces = face_con
	con_coords = bary_con
	con_xy = torch.tensor(con_map_xy, device=device, dtype=dtype)

    # Weights used for the pose prior and the shape prior
	data_weights = [1.] * 5
	body_pose_prior_weights = [404, 404, 57.4, 4.78, 4.78]
	shape_weights = [100, 50, 10, 5, 5]
	coll_loss_weights = [0, 0, 0, 0.01, 1.]
	# bending_prior_weights = [404, 404, 57.4, 4.78, 4.78]

	opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights,
                        }
                        # 'bending_prior_weight': bending_prior_weights

	if interpenetration:
		opt_weights_dict['coll_loss_weight'] = coll_loss_weights

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
        # body model
		body_model.reset_params(body_pose=body_mean_pose)

        # The closure passed to the optimizer
		camera_loss.reset_loss_weights({'data_weight': data_weight})

        # Update the value of the translation of the camera as well as
        # the image center.
		with torch.no_grad():
			camera.translation[:] = init_t.view_as(camera.translation)
			camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

        # Re-enable gradient calculation for the camera translation
		camera.translation.requires_grad = True

		camera_opt_params = [camera.translation, body_model.global_orient]

		camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params)

    	# The closure passed to the optimizer
		fit_camera = monitor.create_fitting_closure(
                camera_optimizer, body_model, camera, cam_need_xy,
                face_cam, bary_cam,
                camera_loss, create_graph=camera_create_graph,
       			vposer=vposer,
                pose_embedding=pose_embedding,
                return_full_pose=False, return_verts=True)
        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
		camera_init_start = time.time()
		if interactive:
			print("Optimize camera params...")
		cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=True,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

		if interactive:
			if torch.cuda.is_available():
				torch.cuda.synchronize()
			tqdm.write('Camera initialization done after {:.4f}'.format(
                    time.time() - camera_init_start))
			tqdm.write('Camera initialization final loss {:.4f}'.format(
                    cam_init_loss_val))

		if check_body_back:
			body_orient = body_model.global_orient.detach().cpu().numpy()
			flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
				cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
			flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

			flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
			orientations = [body_orient, flipped_orient]
		else:
			orientations = [body_model.global_orient.detach().cpu().numpy()]

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

			new_params = defaultdict(global_orient=orient,
                                     body_pose=body_mean_pose)
			body_model.reset_params(**new_params)

			with torch.no_grad():
				pose_embedding.fill_(0)

            # Loss curve
			plt=None
			if show_loss:
				import matplotlib.pyplot as plt
				plt.ion()
				plt.figure()
				plt.title('{}'.format(show_loss_type))

			for opt_idx, curr_weights in enumerate(opt_weights):

				body_params = list(body_model.parameters())

				final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

				final_params.append(pose_embedding)

				body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,)
				body_optimizer.zero_grad()

				curr_weights['data_weight'] = data_weight
				curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])

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
                    vposer=vposer,
                    use_vposer=True,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True, 
                    con_xy=con_xy,
                    con_faces=con_faces, 
                    con_coords=con_coords,
                    )

				if interactive:
					if torch.cuda.is_available():
						torch.cuda.synchronize()
					stage_start = time.time()

				final_loss_val = monitor.run_fitting(
                	body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=True)

				body_params = list(body_model.parameters())

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

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
			result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
			result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
			result['vpose_para'] = pose_embedding.detach().cpu().numpy()

			body_pose = vposer.decode(
                pose_embedding,
                output_type='aa').view(1, -1)
			wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
			body_pose = torch.cat([body_pose, wrist_pose], dim=1)
			body_poses.append(body_pose)
			result['pose_para'] = body_pose.detach().cpu().numpy()

			model_output = body_model(return_verts=True, body_pose=body_pose)

			joints2 = camera(model_output.joints)
			joints3 = model_output.joints.detach().cpu().numpy()
			result['joints3'] = joints3
			result['joints2'] = joints2.detach().cpu().numpy()
			results.append({'loss': final_loss_val,
                            'result': result})


		with open(param_res, 'wb') as result_file:
			if len(results) > 1:
				min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
			else:
				min_idx = 0
			pk.dump(results[min_idx]['result'], result_file, protocol=2)

	model_output = body_model(return_verts=True, body_pose=body_poses[min_idx])
	vertices = model_output.vertices.detach().cpu().numpy().squeeze()

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


def get_cam_need(map_iuv, map_xy):
	# select chest part
	chest_part = [2, 1]
	chest_fron = np.where(map_iuv[:,0] == 2)[0]
	chest_back = np.where(map_iuv[:,0] == 1)[0]

	check_body_back = len(chest_fron)<len(chest_back)

	sel_cam_part = chest_part[0] if not check_body_back \
		else chest_part[1]

	sel_cam_total = map_xy[chest_fron] if sel_cam_part == 2 \
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

	face_cam , bary_cam = calculaterFB(map_uv_cam)

	return edge_idxs, face_cam, bary_cam, map_xy_cam, check_body_back


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
		if part_len == 0:
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
		part_xy_index = []
		for j in range(len(part_map[i])):
			now_xy = part_map[i][j]
			try:
				now_id = np.argwhere((map_xy[:, 0] == now_xy[0]) & \
					(map_xy[:, 1] == now_xy[1]))[0][0]
			except:
				continue
			else:
				part_xy_index.append(now_id)
		part_map_index.append(part_xy_index)
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
	model_params = dict(model_path=model_folder,
						create_body_pose=False,
						create_transl=False,
						dtype=dtype,
						)

	smpl = smplx.create(model_type='smpl', 
						gender=gender, **model_params)

	# load camera
	camera = create_camera(focal_length_x=focal, 
						   focal_length_y=focal,
						   dtype=dtype)

	# load vposer
	vposer, _ = load_vposer(vposer_ckpt,
							vp_model='snapshot')

	# to device
	camera = camera.to(device=device)
	smpl   = smpl.to(device=device)
	vposer = vposer.to(device=device)
	vposer.eval()

	# results floder
	img_fn = osp.join(res_folder, 'images')
	param_fn = osp.join(res_folder, 'params')

	if not osp.exists(img_fn):
		os.makedirs(img_fn)

	if not osp.exists(param_fn):
		os.makedirs(param_fn)

	# load iuv map and origin image
	img_ori = cv2.imread(osp.join(data_folder+'/ori',\
	 img_ori))
	# img_ori = img_ori[:,:int(img_ori.shape[1]/2)+1]
	img_ori = img_ori[...,::-1]/255.0
	# iuv_png = cv2.imread(osp.join('./test_data1/dp', \
	# 	img_dp))
	iuv_png = np.load(osp.join(data_folder+'/dp', \
		img_dp))

	# img_res   = osp.join(img_fn, img_dp.replace('.png', '.png'))
	# param_res = osp.join(param_fn, img_dp.replace('.png', '.pkl'))

	img_res   = osp.join(img_fn, img_dp.replace('.npy', '.png'))
	param_res = osp.join(param_fn, img_dp.replace('.npy', '.pkl'))


	map_iuv, map_xy = get_map_iuv_xy(iuv_png)

	face_con, bary_con, con_map_xy = get_contour_pt_face_bary(\
		iuv_png, map_iuv, map_xy)

	# get estimated camera para points
	edge_idxs, face_cam, bary_cam, map_xy_cam, check_body_back = \
		get_cam_need(map_iuv, map_xy)

	# sample area and get face index, barycentric
	face_index, bary_coords, fit_map_xy = sample_fit_pt_face_bary(\
		map_iuv, map_xy)

	iuv_fitting(body_model=smpl,
				camera=camera,
				vposer=vposer,
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
				con_map_xy=con_map_xy,
				face_con=face_con,
				bary_con=bary_con,
				)

	if interactive:
		tot_time = time.time()
		print('Total opt time: {}'.format(tot_time-app_start))


if __name__ == '__main__':
	app_start = time.time()

	img_ori_p = data_folder+'/ori'
	img_dp_p = data_folder+'/dp'
	img_ori_files = sorted(os.listdir(img_ori_p))
	img_dp_files = sorted(os.listdir(img_dp_p))
	for i in range(len(img_ori_files)):
		img_ori = img_ori_files[i]
		img_dp  = img_dp_files[i]
		main(app_start, img_ori, img_dp)