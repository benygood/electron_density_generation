from utils.data_preprocess import ElectronDensityDirDataset
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import utils.provider as provider
import importlib
import shutil
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import utils.dataset_collate_ignore_none as clfn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('EDGen')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: vn_dgcnn_cls]',
                        choices=['pointnet_cls', 'vn_pointnet_cls', 'dgcnn_cls', 'vn_dgcnn_cls'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training [default: 32]')
    parser.add_argument('--epoch', default=250, type=int, help='Number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate [default: 1e-4]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Pptimizer for training [default: SGD]')
    parser.add_argument('--gpu', type=int, default=[0], help='Specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--log_dir', type=str, default='vn_dgcnn/aligned',
                        help='Experiment root [default: vn_dgcnn/aligned]')
    parser.add_argument('--normal', action='store_true', default=True,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--rot', type=str, default='aligned',
                        help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--pooling', type=str, default='mean', help='VNN only: pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--n_knn', default=20, type=int,
                        help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    return parser.parse_args()


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data

        trot = None
        if args.rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points)

        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(args.gpu))
    torch.cuda.set_device(args.gpu)  # 作用相当于CUDA_VISIBLE_DEVICES命令，修改环境变量
    dist.init_process_group(backend='nccl')  # 设备间通讯通过后端backend实现，GPU上用nccl，CPU上用gloo

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath('gen_mol')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = '/home/jovyan/data/xtb_enamine'

    TRAIN_DATASET = ElectronDensityDirDataset(DATA_PATH, split='train', sample_npoints=args.num_point)
    VALID_DATASET = ElectronDensityDirDataset(DATA_PATH, split='valid', sample_npoints=args.num_point)
    TEST_DATASET = ElectronDensityDirDataset(DATA_PATH,  split='test', sample_npoints=args.num_point)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, collate_fn = clfn.collate_ignore_none, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=4)
    validDataLoader = torch.utils.data.DataLoader(VALID_DATASET, collate_fn = clfn.collate_ignore_none, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, collate_fn = clfn.collate_ignore_none, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(last_npoint=10, atom_num_per_last_point = 10, atom_type_num = 10, normal_channel=args.normal)
    criterion = MODEL.get_loss()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        classifier = torch.nn.DataParallel(classifier)
        criterion = torch.nn.DataParallel(criterion)
    if torch.cuda.is_available():
        classifier.cuda()
        criterion.cuda()
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate * 100,
            momentum=0.9,
            weight_decay=args.decay_rate
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_gen_type_correct = []
    mean_target_type_correct = []
    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            _, points, target = data

            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(points.shape[0]))
            if trot is not None:
                points = trot.transform_points(points)

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)

            points = points.transpose(2, 1)
            if torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            center_coords, coords, types = classifier(points)
            loss, gen_type_loss, target_type_loss, emd_loss, gen_type_correct,  target_type_correct = criterion(center_coords, coords, types, target)
            mean_gen_type_correct.append(gen_type_correct.cpu())
            mean_target_type_correct.append(target_type_correct.cpu())
            loss.sum().backward()
            optimizer.step()
            global_step += 1

        log_string('Train Instance gen type Accuracy: %f' % np.mean(mean_gen_type_correct))
        log_string('Train Instance target type Accuracy: %f' % np.mean(mean_target_type_correct))

        # with torch.no_grad():
        #     instance_acc, class_acc = test(classifier.eval(), testDataLoader)
        #
        #     if (instance_acc >= best_instance_acc):
        #         best_instance_acc = instance_acc
        #         best_epoch = epoch + 1
        #
        #     if (class_acc >= best_class_acc):
        #         best_class_acc = class_acc
        #     log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        #     log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
        #
        #     if (instance_acc >= best_instance_acc):
        #         logger.info('Save model...')
        #         savepath = str(checkpoints_dir) + '/best_model.pth'
        #         log_string('Saving at %s' % savepath)
        #         state = {
        #             'epoch': best_epoch,
        #             'instance_acc': instance_acc,
        #             'class_acc': class_acc,
        #             'model_state_dict': classifier.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #         }
        #         torch.save(state, savepath)
        #     global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
