from email.mime import base
import numpy as np
import open3d as o3d
import os
from main import train_one_epoch
import torch
import torch.optim as optim
from util import npmat2euler
import gc
import  wandb

# basefilename = "/home/akshay/Downloads/2011_09_26_downsampled/2011_09_26_downsampled/2011_09_26_drive_0001_sync/velodyne_points/data/"
# for filename in sorted(os.listdir(basefilename)):
#     fullfilename = os.path.join(basefilename, filename)
#     pcd = o3d.io.read_point_cloud(fullfilename)
#     print(np.asarray(pcd.points).shape)
#     o3d.visualization.draw_geometries([pcd])

def train_slam(args, 
               net, 
               train_loader, 
               test_loader=None, 
               boardio=None, 
               textio=None, 
               scaler=None, 
               preload=True,
               model_path="/home/akshay/Deep_Closest_Point/model_idgcn_full.pth",
               filename_for_folder=None):
    if preload:
        if net.load_state_dict(torch.load(model_path), strict=False):
            print("Loaded from ", filename_for_folder)
        else:
            print("Not loaded pre-trained")
    print("Training slam")
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr*0.1, weight_decay=1e-4)
        #opt.load_state_dict(torch.load('opt.pth'))
    #scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    #scheduler = ReduceLROnPlateau(opt, mode='min', factor='.1', patience=

    for epoch in range(250, args.epochs+250, 1):
        torch.cuda.empty_cache()
        #scheduler.step()
        train_loss, train_cycle_loss, \
        train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, \
        train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
        train_translations_ba_pred, train_eulers_ab, train_eulers_ba = train_one_epoch(args, net, train_loader, opt, epoch, scaler)
        train_rmse_ab = np.sqrt(train_mse_ab)
        train_rmse_ba = np.sqrt(train_mse_ba)

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))


        train_rotations_ba_pred_euler = npmat2euler(train_rotations_ba_pred, 'xyz')
        train_r_mse_ba = np.mean((train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)) ** 2)
        train_r_rmse_ba = np.sqrt(train_r_mse_ba)
        train_r_mae_ba = np.mean(np.abs(train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)))
        train_t_mse_ba = np.mean((train_translations_ba - train_translations_ba_pred) ** 2)
        train_t_rmse_ba = np.sqrt(train_t_mse_ba)
        train_t_mae_ba = np.mean(np.abs(train_translations_ba - train_translations_ba_pred))

        wandb.log({'rotation/MSE/a-b/': train_r_mse_ab, 'translation/MSE': train_t_mse_ab,
                   'rotation/MSE/b-a/': train_r_mse_ab}, step=epoch)
                   
        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss:, %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, train_cycle_loss, train_mse_ab, train_rmse_ab, train_mae_ab, train_r_mse_ab,
                         train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, train_mse_ba, train_rmse_ba, train_mae_ba, train_r_mse_ba, train_r_rmse_ba,
                         train_r_mae_ba, train_t_mse_ba, train_t_rmse_ba, train_t_mae_ba))

        torch.save(net.state_dict(), filename_for_folder + "_model.pth")
        torch.save(opt.state_dict(), filename_for_folder + '_opt.pth')

        gc.collect()

if __name__ == '__main__':
    pass