import torch
import numpy as np
from model import DCP
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N', choices=['dcp'], help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='idgcn', metavar='N', choices=['pointnet', 'dgcnn', 'idgcn'])
    parser.add_argument('--pointer', type=str, default='identity', metavar='N', choices=['identity', 'transformer'])
    parser.add_argument('--head', type=str, default='svd', metavar='N', choices=['mlp', 'svd', ])
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N', help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='slam', choices=['modelnet40', 'modelnet10', "slam"], metavar='N')
    parser.add_argument('--factor', type=float, default=4, metavar='N', help='Divided factor for rotations')
    parser.add_argument('--agg_fun_name', type=str, default='attention', metavar='N', help='agg functions to use')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N', help='Whether to use cycle consistency')
    
    args = parser.parse_args()
    args.num_points = 1024

    model = DCP(args)
    model.load_state_dict(torch.load('/home/akshay/Deep_Closest_Point/slam_model.pth'), strict=False)
    model = model.eval()
    print(model)

    #### target current cloud
    #### src previous cloud
