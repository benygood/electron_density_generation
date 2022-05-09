import os
import sys
import argparse

import numpy as np

from utils.data_preprocess import ElectronDensityDirDataset
from utils.to_pdb import produce_pdb_ligand_with_confidence
import utils.data_preprocess as dp
import importlib
import torch
import traceback as tb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('EDGen')
    parser.add_argument('--model', default='vn_pointnet2_cls_ssg', help='Model name [default: vn_dgcnn_cls]',
                        choices=['pointnet_cls', 'vn_pointnet_cls', 'dgcnn_cls', 'vn_dgcnn_cls'])
    parser.add_argument('--gpu', type=str, default='-1', help='Specify gpu device [default: 0]')

    parser.add_argument('--ed', help='input electron density file. npy format')
    parser.add_argument('--checkpoint_path', help='model checkpoint path')
    parser.add_argument('--normal', action='store_true', default=True,
                        help='Whether to use normal information [default: False]')
    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(last_npoint=10, atom_num_per_last_point=10, atom_type_num=10,
                                 normal_channel=args.normal)
    try:
        checkpoint = torch.load(args.checkpoint_path,map_location=torch.device('cpu'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(e.args)
        print(tb.format_exc())
    classifier.eval()

    ds = ElectronDensityDirDataset("")
    ed_path = args.ed
    dir_path = os.path.dirname(ed_path)
    filename = os.path.basename(ed_path)
    pdb_path = os.path.join(dir_path, "{}.pdb".format(filename[:-4]))
    pdb_path_out = os.path.join(dir_path, "{}_out.pdb".format(filename[:-4]))
    _, points, target = ds.get(ed_path,pdb_path)

    center_coords, coords, types = classifier(torch.Tensor(points[np.newaxis, :, :]))
    symbols = dp.atom_type_config_arr[types.max(-1)[1].view(-1)]
    coords = np.array(coords.view(-1,3).tolist())*10
    coords = np.hstack([coords,symbols[:,np.newaxis]])
    produce_pdb_ligand_with_confidence(coords, pdb_path_out[:-4])

    shape = center_coords.shape
    symbols = np.array(['C']*shape[-2])
    coords = np.array(center_coords.view(-1,3).tolist())*10
    coords = np.hstack([coords, symbols[:, np.newaxis]])
    produce_pdb_ligand_with_confidence(coords, "{}_centers".format(pdb_path_out[:-4]))
if __name__ == '__main__':
    args = parse_args()
    main(args)