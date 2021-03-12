"""
If code works:
    Author: Nianhong Jiao
else:
    Author: Anonymous
"""

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import shutil
import warnings
warnings.filterwarnings("ignore")
import pickle
import glob
import os
import sys
import numpy as np
import pickle as pk
import time
import os.path as osp
import smplx
import cv2
from tqdm import tqdm
from collections import defaultdict
import PIL.Image as pil_img

from my_use.camera import create_camera
from my_use import fitting
from my_use.prior import create_prior
from human_body_prior.tools.model_loader import load_vposer
from optimizers import optim_factory
from my_use.utils import render
from utils.daz import DazModel

import torch

focal = 5000 * 28. / 234.7000
iter_stage = 5
batch_size = 1
dtype = torch.float32
# gender       = 'neutral'
gender = 'female'
# gender       = 'male'
visualize = False


vposer_ckpt = './vposer_v1_0'
model_folder = '/home/jnh/workspace/workspace_bak/smplx/download/models'

part_segm_path = ''
interpenetration = False
interactive = False

show_loss = False
show_loss_type = 'pro_loss'


data_folder = '/media/jnh/Jnh_T5/ihp_data/daz2smpl_5000_data/mini_data'
res_folder = '/media/jnh/Jnh_T5/ihp_data/daz2smpl_5000_data/smpl_data_test'


def nbd_fitting(body_model, camera, vposer, img, img_res, param_res,
                index_smpl_346, target_vertex, go_pose, pose_encode, smpl_init_transl, device):

    target_vertex_tensor = torch.from_numpy(target_vertex).to(device)
    index_smpl_346_tensor = torch.from_numpy(index_smpl_346).long().to(device)
    # mean pose
    pose_embedding = torch.from_numpy(pose_encode).to(device)
    pose_embedding.requires_grad = True

    init_t = torch.zeros((1, 3)).to(device)
    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      depth_loss_weight=1e2,
                                      dtype=dtype,
                                      ).to(device=device)

    camera_loss.trans_estimation[:] = init_t

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None

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

    body_mean_pose = torch.from_numpy(pose_encode).to(device)

    # Weights used for the pose prior and the shape prior
    data_weights = [1.] * 5
    body_pose_prior_weights = [404, 404, 57.4, 4.78, 4.78]
    shape_weights = [100, 50, 10, 5, 5]
    # bending_prior_weights = [404, 404, 57.4, 4.78, 4.78]

    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights,
                        }
    # 'bending_prior_weight': bending_prior_weights

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

        # img = torch.tensor(img, dtype=dtype)
        # H, W, _ = img.shape
        H, W = 512, 512

        data_weight = 1000 / H
        # Reset the parameters to estimate the initial translation of the
        # body model
        body_model.reset_params(body_pose=body_mean_pose,
                                global_orient=torch.from_numpy(go_pose).to(device),
                                transl=torch.from_numpy(smpl_init_transl).to(device))

        # The closure passed to the optimizer
        camera_loss.reset_loss_weights({'data_weight': data_weight})

        # Update the value of the translation of the camera as well as
        # the image center.
        with torch.no_grad():
            camera.translation[:] = init_t.view_as(camera.translation)
            camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

        # Re-enable gradient calculation for the camera translation
        camera.translation.requires_grad = False

        camera_opt_params = [body_model.global_orient, body_model.transl]

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params)

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, camera,
            target_vertex_tensor, index_smpl_346_tensor,
            loss=camera_loss, create_graph=camera_create_graph,
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

        if 0:
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
        results=[]

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

            # with torch.no_grad():
            #     pose_embedding.fill_(0)

            # Loss curve
            plt = None
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
                    final_params, )
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                        3.17 * curr_weights['body_pose_weight'])

                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera,
                    target_vertex_tensor, index_smpl_346_tensor,
                    plt=plt,
                    show_loss=show_loss,
                    show_loss_type=show_loss_type,
                    loss=loss, create_graph=body_create_graph,
                    vposer=vposer,
                    use_vposer=True,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True,
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

        # with open(param_res, 'wb') as result_file:
        #     if len(results) > 1:
        #         min_idx = (0 if results[0]['loss'] < results[1]['loss']
        #                    else 1)
        #     else:
        #         min_idx = 0
        #     pk.dump(results[min_idx]['result'], result_file, protocol=2)

    # model_output = body_model(return_verts=True, body_pose=body_poses[0])
    # vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    # smpl_face = body_model.faces

    # import trimesh
    # smpl_mesh = trimesh.Trimesh(vertices, smpl_face)
    # smpl_mesh.export('./smpl.obj')

    color = render(vertices, smpl_face, camera,
                   focal=focal, depth_or_color='color', H=H, W=W)
    color = color.astype(np.float32) / 255.0

    # img = img.detach().cpu().numpy()
    # origin_img = img.astype(np.float32) / 255
    # valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis].astype(np.float32) * 1.0
    # show_img1 = origin_img * (1 - valid_mask) + color[:, :, :3] * valid_mask

    # j2d = joints2.detach().cpu().numpy()[0]
    # drawPts2img(j2d, show_img1)
    # exit()

    # cv2.imshow('test', show_img1)
    # cv2.waitKey()
    # img = pil_img.fromarray((show_img1 * 255).astype(np.uint8))
    # img.save(img_res)

    return results[0]


def cvShow(img):
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 640, 480)
    cv2.imshow('show', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def drawPts2img(pts, img):
    import copy
    draw_img = copy.deepcopy(img)
    for i in range(pts.shape[0]):
        cv2.circle(draw_img, (int(pts[i, 0]), int(pts[i, 1])), 3, (0, 0, 255, 255), -1)
        # cv2.putText(draw_img, str(i), (int(pts[i, 0] + 2), int(pts[i, 1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    cvShow(draw_img)


def main(img_ori, pkl_ori, device):
    daz = DazModel('./model')
    with open(pkl_ori, 'rb') as rot:
        para_data = pickle.load(rot)

        now_pose = np.array(para_data['daz_female_pose']).astype(np.float32).reshape([-1, 3])
        now_beta = np.array(para_data['daz_female_beta']).astype(np.float32).reshape([1, -1])[:, :21]
        now_tran = np.array(para_data['daz_female_transl']).astype(np.float32)
        init_pose = torch.from_numpy(now_pose.reshape([1, -1, 3]).astype(np.float32))
        init_beta = torch.from_numpy(now_beta.reshape([1, -1]).astype(np.float32))
        init_tran = torch.from_numpy(now_tran.reshape([1, -1]).astype(np.float32))

    body_pose = init_pose[:, 1:].detach().clone()
    global_orient = init_pose[:, :1].detach().clone()
    betas = init_beta.detach().clone()
    vers, model_joints, _ = daz(global_orient=global_orient,
                                body_pose=body_pose,
                                betas=betas, transl=init_tran)
    pred_vertices = vers[0].detach().cpu().numpy()
    # import trimesh
    # now_mesh = trimesh.Trimesh(pred_vertices, daz.faces)
    # now_mesh.export('./daz.obj')
    # now_mesh.show()

    # model initial
    model_params = dict(model_path=model_folder,
                        create_body_pose=False,
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
    smpl = smpl.to(device=device)
    vposer = vposer.to(device=device)
    vposer.eval()

    # results folder
    img_fn = osp.join(res_folder, 'images')
    ren_fn = osp.join(res_folder, 'renders')
    param_fn = osp.join(res_folder, 'params')

    # os.makedirs(img_fn, exist_ok=True)
    # os.makedirs(ren_fn, exist_ok=True)
    # os.makedirs(param_fn, exist_ok=True)
    # shutil.copy(img_ori, img_fn)

    # img = cv2.imread(img_ori)
    img=None
    img_name = os.path.basename(img_ori)

    # img_res = osp.join(ren_fn, img_name)
    img_res = img_ori.replace('.jpg', '.png')
    param_res = osp.join(param_fn, img_name.replace('.jpg', '.pkl'))
    # if os.path.exists(param_res):
    #     return None

    # prepare fitting data, init pose, init transl, target vertex
    index_smpl_346 = np.loadtxt('./my_use/sample_index_346.txt').astype(np.int32)
    index_daz_346 = np.loadtxt('./my_use/sample_index_daz_346.txt').astype(np.int32)
    target_vertex = pred_vertices[index_daz_346]

    daz2smpl_pose_index = np.array([2, 18, 35, 4, 20, 36, 5, 21, 37, 7, 23, 88, 38, 63, 90, 39, 64, 41, 66, 43, 68]).astype(np.uint8)

    smpl_init_pose = now_pose[daz2smpl_pose_index]
    smpl_init_pose = torch.from_numpy(smpl_init_pose.reshape(1, -1)).to(device)
    go_pose = now_pose[0]
    pose_encode = vposer.encode(smpl_init_pose)
    pose_encode = (pose_encode.loc).detach().cpu().numpy()
    smpl_init_transl = now_tran.copy()

    now_data = nbd_fitting(smpl, camera, vposer, img, img_res, param_res,
                index_smpl_346, target_vertex, go_pose, pose_encode, smpl_init_transl, device)


    para_data['smpl_data'] = now_data
    pickle.dump(para_data, open(pkl_ori, 'wb'))


def mul_pro(all_list, gpu_id=None):
    device = torch.device('cuda:%d'%gpu_id)
    for i in tqdm(range(len(all_list))):

        img_ori = all_list[i]
        if not os.path.exists(img_ori):
            continue
        pkl_ori = img_ori.replace('.jpg', '.pickle')
        if not os.path.exists(pkl_ori):
            continue
        # print(img_ori)
        try:
            main(img_ori, pkl_ori, device)
        except:
            pass


if __name__ == '__main__':

    daz_ori_path = data_folder
    daz_data_list = glob.glob(daz_ori_path + '/*.jpg')
    import random
    random.shuffle(daz_data_list)

    gpu_list = [0, 1]
    thread_num = 2 * len(gpu_list)
    split_list = np.array_split(daz_data_list, thread_num)

    from multiprocessing import Process
    process_list = []
    for idx in range(thread_num):
        gpu_id = gpu_list[idx % len(gpu_list)]
        p = Process(target=mul_pro, args=(split_list[idx], gpu_id))
        process_list.append(p)
        p.start()

    for idx in range(len(process_list)):
        process_list[idx].join()






