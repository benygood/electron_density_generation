import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from pointnet2_utils import square_distance


class get_model(nn.Module):
    def __init__(self, last_npoint=10, atom_num_per_last_point = 10, atom_type_num = 10, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 4 if normal_channel else 3
        self.normal_channel = normal_channel
        self.atom_num_per_last_point = atom_num_per_last_point
        self.atom_type_num = atom_type_num
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=60, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.3, nsample=20, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=last_npoint, radius=0.5, nsample=8, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.conv_coord = nn.Conv1d(512, atom_num_per_last_point*3*16, 1)
        self.conv_coord_bn = nn.BatchNorm1d(atom_num_per_last_point*3*16)
        self.conv_type = nn.Conv1d(512, atom_num_per_last_point*16, 1)
        self.conv_type_bn = nn.BatchNorm1d(atom_num_per_last_point*16)
        self.fc1 = nn.Linear(16, 1)
        self.fc2 = nn.Linear(16, atom_type_num)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #todo deal with gnn for l3 to get global information

        #fixme:  how to generate multi points for every l3 point: here use mlp and reshape;
        # maybe use rnn or transformer is better
        coords_emb = F.relu(self.conv_coord_bn(self.conv_coord(l3_points))).permute(0, 2, 1)
        coords_emb = coords_emb.reshape(B,-1, self.atom_num_per_last_point, 3, 16)
        atoms_emb = F.relu(self.conv_type_bn(self.conv_type(l3_points))).permute(0,2,1)
        atoms_emb = atoms_emb.reshape(B,-1, self.atom_num_per_last_point, 16)
        coords = self.fc1(coords_emb).squeeze()
        # add abs center coordinates
        l3_xyz = l3_xyz.permute(0,2,1)
        l3_xyz_tile = l3_xyz.repeat(1, 1, self.atom_num_per_last_point)
        l3_xyz_tile = l3_xyz_tile.view(B, -1, self.atom_num_per_last_point, 3)
        coords = coords + l3_xyz_tile
        atoms = F.log_softmax(self.fc2(atoms_emb), -1)
        return l3_xyz, coords, atoms



class get_loss(nn.Module):
    def __init__(self, atom_num_per_last_point = 10 ):
        super(get_loss, self).__init__()
        self.atom_num_per_last_point = atom_num_per_last_point

    def forward(self, center_coords, coords, types, target):
        #todo: use EMD loss
        # center_coords & target 's distance
        center_num = center_coords.shape[1]
        type_num = types.shape[-1]
        dist = square_distance(center_coords, target[:,:,:-1])
        vals, inds = dist.sort()

        inds_topN = inds[:,:,:self.atom_num_per_last_point]
        inds_topN_ex = inds_topN.unsqueeze(-1)
        inds_topN_ex = inds_topN_ex.repeat(1,1,1,4)
        target_ex = target.unsqueeze(1)
        target_ex = target_ex.repeat(1,center_num,1,1)
        target_ex = target_ex.gather(2, inds_topN_ex)
        dist_local = square_distance(coords, target_ex[:,:,:,:-1])
        inds_gen_local =  dist_local.argmin(-1, keepdim = True)
        inds_target_local = dist_local.argmin(-2, keepdim = True)
        dist_emd_gen_local = dist_local.gather(-1, inds_gen_local)
        dist_emd_target_local =  dist_local.gather(-2, inds_target_local)
        target_ex_type = target_ex[:,:,:,-1].long()
        target_ex_type_for_gen = target_ex_type.gather(-1, inds_gen_local.squeeze())
        gen_type_loss = F.nll_loss(types, target_ex_type_for_gen)
        gen_type_correct = (types.max(-1)[1] == target_ex_type_for_gen).float().mean()
        inds_target_local_ex = inds_target_local.squeeze().unsqueeze(-1).repeat(1,1,1,type_num)
        gen_types_for_target = types.gather(-2, inds_target_local_ex)
        target_type_loss = F.nll_loss(gen_types_for_target, target_ex_type)
        target_type_correct = (gen_types_for_target.max(-1)[1] == target_ex_type).float().mean()
        emd_loss = dist_emd_gen_local.mean() + dist_emd_target_local.mean()
        loss = gen_type_loss + target_type_loss + emd_loss
        return loss, gen_type_loss, target_type_loss, emd_loss, gen_type_correct,  target_type_correct
