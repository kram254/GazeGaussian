import os
import torch
from torch import nn
import numpy as np
import kaolin
import tqdm
import time

def knn_points(p1, p2, K=1):
    dist = torch.cdist(p1, p2)
    dists, idx = torch.topk(dist, K, dim=-1, largest=False)
    return dists, idx, None

def so3_exponential_map(log_rot):
    theta = torch.norm(log_rot, dim=-1, keepdim=True)
    theta = torch.clamp(theta, min=1e-8)
    w = log_rot / theta
    wx = torch.zeros(log_rot.shape[0], 3, 3, device=log_rot.device)
    wx[:, 0, 1] = -w[:, 2]
    wx[:, 1, 0] = w[:, 2]
    wx[:, 0, 2] = w[:, 1]
    wx[:, 2, 0] = -w[:, 1]
    wx[:, 1, 2] = -w[:, 0]
    wx[:, 2, 1] = w[:, 0]
    I = torch.eye(3, device=log_rot.device).unsqueeze(0)
    R = I + torch.sin(theta).unsqueeze(-1) * wx + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(wx, wx)
    return R

def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(rotation_matrices):
    batch_dim = rotation_matrices.shape[:-2]
    m00, m01, m02 = rotation_matrices[..., 0, 0], rotation_matrices[..., 0, 1], rotation_matrices[..., 0, 2]
    m10, m11, m12 = rotation_matrices[..., 1, 0], rotation_matrices[..., 1, 1], rotation_matrices[..., 1, 2]
    m20, m21, m22 = rotation_matrices[..., 2, 0], rotation_matrices[..., 2, 1], rotation_matrices[..., 2, 2]
    
    trace = m00 + m11 + m22
    
    q = torch.zeros((*batch_dim, 4), dtype=rotation_matrices.dtype, device=rotation_matrices.device)
    
    q[..., 0] = torch.sqrt(torch.clamp(1.0 + trace, min=0.0)) / 2.0
    q[..., 1] = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) / 2.0
    q[..., 2] = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)) / 2.0
    q[..., 3] = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)) / 2.0
    
    q[..., 1] *= torch.sign(m21 - m12)
    q[..., 2] *= torch.sign(m02 - m20)
    q[..., 3] *= torch.sign(m10 - m01)
    
    return q

from models.MLP import MLP
from models.PositionalEmbedding import get_embedder
from configs.meshhead_options import BaseOptions

from utils.dmtet_utils import marching_tetrahedra

class MeshHeadModule(nn.Module):
    def __init__(self, opt: BaseOptions, init_landmarks_3d_neutral):
        super(MeshHeadModule, self).__init__()
        
        self.geo_mlp = MLP(opt.geo_mlp, last_op=nn.Tanh())
        self.shape_color_mlp = MLP(opt.shape_color_mlp, last_op=nn.Sigmoid())
        self.pose_color_mlp = MLP(opt.pose_color_mlp, last_op=nn.Sigmoid()) 
        self.eye_color_mlp = MLP(opt.eye_color_mlp, last_op=nn.Sigmoid())

        self.shape_deform_mlp = MLP(opt.shape_deform_mlp, last_op=nn.Tanh())
        self.pose_deform_mlp = MLP(opt.pose_deform_mlp, last_op=nn.Tanh())
        self.is_rotate_eye = opt.is_rotate_eye
        if opt.is_rotate_eye:
            self.eye_deform_mlp = MLP(opt.eye_deform_mlp_rotate, last_op=nn.functional.normalize)
        else:
            self.eye_deform_mlp = MLP(opt.eye_deform_mlp, last_op=nn.Tanh())




        self.landmarks_3d_neutral = nn.Parameter(init_landmarks_3d_neutral)

        self.pos_embedding, _ = get_embedder(opt.pos_freq)
        self.gaze_embedding, _ = get_embedder(opt.gaze_freq)

        self.model_bbox = opt.model_bbox
        self.dist_threshold_near = opt.dist_threshold_near
        self.dist_threshold_far = opt.dist_threshold_far
        self.deform_scale = opt.deform_scale

        tets_data_path = 'configs/config_models/tets_data.npz'
        if not os.path.exists(tets_data_path):
            raise FileNotFoundError(
                f"Critical file missing: {tets_data_path}\n"
                "Please download it from HuggingFace:\n"
                "  cd configs\n"
                "  wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/config_models.zip\n"
                "  unzip config_models.zip\n"
                "Or follow the README instructions."
            )
        tets_data = np.load(tets_data_path)
        self.register_buffer('tet_verts', torch.from_numpy(tets_data['tet_verts']))
        self.register_buffer('tets', torch.from_numpy(tets_data['tets']))
        self.grid_res = 128

        if opt.subdivide:
            self.subdivide()

    def geometry(self, geo_input):
        pred = self.geo_mlp(geo_input)
        return pred

    def shape_color(self, color_input):
        verts_color = self.shape_color_mlp(color_input)
        return verts_color
    
    def pose_color(self, color_input):
        verts_color = self.pose_color_mlp(color_input)
        return verts_color

    def eye_color(self, color_input):
        verts_color = self.eye_color_mlp(color_input)
        return verts_color
    
    def shape_deform(self, deform_input):
        deform = self.shape_deform_mlp(deform_input)
        return deform
    
    def pose_deform(self, deform_input):
        deform = self.pose_deform_mlp(deform_input)
        return deform

    def eye_deform(self, deform_input):
        deform = self.eye_deform_mlp(deform_input)
        return deform
        
    def get_landmarks(self):
        return self.landmarks_3d_neutral

    def subdivide(self):
        new_tet_verts, new_tets = kaolin.ops.mesh.subdivide_tetmesh(self.tet_verts.unsqueeze(0), self.tets)
        self.tet_verts = new_tet_verts[0]
        self.tets = new_tets
        self.grid_res *= 2

    def reconstruct(self, data):
        B = data['nl3dmm_para_dict']['shape_code'].shape[0]

        query_pts = self.tet_verts.unsqueeze(0).repeat(B, 1, 1)
        geo_input = self.pos_embedding(query_pts).permute(0, 2, 1)

        pred = self.geometry(geo_input)

        sdf, deform, features = pred[:, :1, :], pred[:, 1:4, :], pred[:, 4:, :]
        sdf = sdf.permute(0, 2, 1)
        features = features.permute(0, 2, 1)
        verts_deformed = (query_pts + torch.tanh(deform.permute(0, 2, 1)) / self.grid_res)
        verts_list, features_list, faces_list = marching_tetrahedra(verts_deformed, features, self.tets, sdf)

        data['verts0_list'] = verts_list
        data['faces_list'] = faces_list

        verts_batch = []
        verts_features_batch = []
        num_pts_max = 0
        for b in range(B):
            if verts_list[b].shape[0] > num_pts_max:
                num_pts_max = verts_list[b].shape[0]
            
        for b in range(B):
            verts_batch.append(torch.cat([verts_list[b], torch.zeros([num_pts_max - verts_list[b].shape[0], verts_list[b].shape[1]], device=verts_list[b].device)], 0))
            verts_features_batch.append(torch.cat([features_list[b], torch.zeros([num_pts_max - features_list[b].shape[0], features_list[b].shape[1]], device=features_list[b].device)], 0))
        verts_batch = torch.stack(verts_batch, 0)
        verts_features_batch = torch.stack(verts_features_batch, 0)

        dists, idx, _ = knn_points(verts_batch, data['landmarks_3d_neutral'])
        shape_weights = torch.clamp((self.dist_threshold_far - dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)
        pose_weights = 1 - shape_weights

        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))
        eye_indices = left_eye_indices + right_eye_indices
        dists, idx, _ = knn_points(verts_batch, data['landmarks_3d_neutral'][:, eye_indices])
        eye_weights = torch.clamp((self.dist_threshold_far - dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)

        if self.is_rotate_eye:
            left_dists, left_idx, _ = knn_points(verts_batch, data['landmarks_3d_neutral'][:, left_eye_indices])
            right_dists, right_idx, _ = knn_points(verts_batch, data['landmarks_3d_neutral'][:, right_eye_indices])
            left_eye_weights = torch.clamp((self.dist_threshold_far - left_dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)
            right_eye_weights = torch.clamp((self.dist_threshold_far - right_dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)

        shape_color_input = torch.cat([verts_features_batch.permute(0, 2, 1), data['nl3dmm_para_dict']['shape_code'].unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        verts_color_batch = self.shape_color(shape_color_input).permute(0, 2, 1) * shape_weights
        
        pose_color_input = torch.cat([verts_features_batch.permute(0, 2, 1), self.pos_embedding(data['nl3dmm_para_dict']['pose']).unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        verts_color_batch = verts_color_batch + self.pose_color(pose_color_input).permute(0, 2, 1) * pose_weights

        eye_color_input = torch.cat([verts_features_batch.permute(0, 2, 1), self.pos_embedding(data['nl3dmm_para_dict']['pitchyaw']).unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        verts_color_batch = verts_color_batch + self.eye_color(eye_color_input).permute(0, 2, 1) * eye_weights


        shape_deform_input = torch.cat([self.pos_embedding(verts_batch).permute(0, 2, 1), data['nl3dmm_para_dict']['shape_code'].unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        shape_deform = self.shape_deform(shape_deform_input).permute(0, 2, 1)
        verts_batch = verts_batch + shape_deform * shape_weights * self.deform_scale

        pose_deform_input = torch.cat([self.pos_embedding(verts_batch).permute(0, 2, 1), self.pos_embedding(data['nl3dmm_para_dict']['pose']).unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        pose_deform = self.pose_deform(pose_deform_input).permute(0, 2, 1)
        verts_batch = verts_batch + pose_deform * pose_weights * self.deform_scale
        if self.is_rotate_eye:
            left_mask = left_eye_weights.squeeze(-1) > 0.0
            left_eye_deform_input = torch.cat([self.pos_embedding(verts_batch[left_mask].mean(dim=0, keepdim=True)), self.pos_embedding(data['nl3dmm_para_dict']['pitchyaw'])], 1).unsqueeze(-1)
            left_eye_deform = self.eye_deform(left_eye_deform_input).permute(0, 2, 1)
            left_rotation_matrices = quaternion_to_matrix(left_eye_deform).expand(-1, left_mask.sum(), -1, -1)
            verts_batch_expanded = verts_batch.unsqueeze(-1)
            left_eye_deform = torch.matmul(left_rotation_matrices, verts_batch_expanded[left_mask]).squeeze(-1)
            verts_batch[left_mask] = left_eye_deform

            right_mask = right_eye_weights.squeeze(-1) > 0.0
            right_eye_deform_input = torch.cat([self.pos_embedding(verts_batch[right_mask].mean(dim=0, keepdim=True)), self.pos_embedding(data['nl3dmm_para_dict']['pitchyaw'])], 1).unsqueeze(-1)
            right_eye_deform = self.eye_deform(right_eye_deform_input).permute(0, 2, 1)
            right_rotation_matrices = quaternion_to_matrix(right_eye_deform).expand(-1, right_mask.sum(), -1, -1)
            verts_batch_expanded = verts_batch.unsqueeze(-1)
            right_eye_deform = torch.matmul(right_rotation_matrices, verts_batch_expanded[right_mask]).squeeze(-1)
            verts_batch[right_mask] = right_eye_deform
        else:
            eye_deform_input = torch.cat([self.pos_embedding(verts_batch).permute(0, 2, 1), self.pos_embedding(data['nl3dmm_para_dict']['pitchyaw']).unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
            eye_deform = self.eye_deform(eye_deform_input).permute(0, 2, 1)
            verts_batch = verts_batch + eye_deform * eye_weights * self.deform_scale


        R = so3_exponential_map(data['nl3dmm_para_dict']['pose'][:, :3])
        T = data['nl3dmm_para_dict']['pose'][:, None, 3:]
        S = data['nl3dmm_para_dict']['scale'][:, :, None]
        verts_batch = torch.bmm(verts_batch * S, R.permute(0,2,1)) + T
        
        data['shape_deform'] = shape_deform
        data['pose_deform'] = pose_deform
        data['verts_list'] = [verts_batch[b, :verts_list[b].shape[0], :] for b in range(B)]
        data['verts_color_list'] = [verts_color_batch[b, :verts_list[b].shape[0], :] for b in range(B)]
        return data
    
    def reconstruct_neutral(self):
        query_pts = self.tet_verts.unsqueeze(0)
        geo_input = self.pos_embedding(query_pts).permute(0, 2, 1)

        pred = self.geometry(geo_input)

        sdf, deform, features = pred[:, :1, :], pred[:, 1:4, :], pred[:, 4:, :]
        sdf = sdf.permute(0, 2, 1)
        features = features.permute(0, 2, 1)
        verts_deformed = (query_pts + torch.tanh(deform.permute(0, 2, 1)) / self.grid_res)
        verts_list, features_list, faces_list = marching_tetrahedra(verts_deformed, features, self.tets, sdf)

        data = {}
        data['verts'] = verts_list[0]
        data['faces'] = faces_list[0]
        data['verts_feature'] = features_list[0]
        return data
    
    def query_sdf(self, data):
        query_pts = data['query_pts']

        geo_input = self.pos_embedding(query_pts).permute(0, 2, 1)

        pred = self.geometry(geo_input)
        sdf = pred[:, :1, :]
        sdf = sdf.permute(0, 2, 1)

        data['sdf'] = sdf
        return data
    
    def deform(self, data):
        code_info = data['shape_code']
        query_pts = data['query_pts']
        
        geo_input = self.pos_embedding(query_pts).permute(0, 2, 1)

        pred = self.geometry(geo_input)
        sdf, deform = pred[:, :1, :], pred[:, 1:4, :]
        query_pts = (query_pts + torch.tanh(deform).permute(0, 2, 1) / self.grid_res)

        shape_deform_input = torch.cat([self.pos_embedding(query_pts).permute(0, 2, 1), code_info.unsqueeze(-1).repeat(1, 1, query_pts.shape[1])], 1)
        shape_deform = self.shape_deform(shape_deform_input).permute(0, 2, 1)

        deformed_pts = query_pts + shape_deform * self.deform_scale

        data['deformed_pts'] = deformed_pts
        return data
    
    def in_bbox(self, verts, bbox):
        is_in_bbox = (verts[:, :, 0] > bbox[0][0]) & \
                     (verts[:, :, 1] > bbox[1][0]) & \
                     (verts[:, :, 2] > bbox[2][0]) & \
                     (verts[:, :, 0] < bbox[0][1]) & \
                     (verts[:, :, 1] < bbox[1][1]) & \
                     (verts[:, :, 2] < bbox[2][1])
        return is_in_bbox
    
    def pre_train_sphere(self, iter, device):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)

        iter_tq = tqdm.tqdm(range(iter))
        for i in iter_tq:
            query_pts = torch.rand((8, 1024, 3), device=device) * 3 - 1.5
            ref_value  = torch.sqrt((query_pts**2).sum(-1)) - 1.0
            data = {
                'query_pts': query_pts
                }
            data = self.query_sdf(data)
            sdf = data['sdf']
            loss = loss_fn(sdf[:, :, 0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_tq.set_postfix({'Pre-trained MLP loss': loss.item()})