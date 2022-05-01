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
import utils.model_stat as model_stat
import utils.provider as provider
import importlib
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import utils.dataset_collate_ignore_none as clfn
import torch.distributed as dist
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


DEBUG = False

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('EDGen')
    parser.add_argument('--model', default='vn_pointnet2_cls_ssg', help='Model name [default: vn_dgcnn_cls]',
                        choices=['pointnet_cls', 'vn_pointnet_cls', 'dgcnn_cls', 'vn_dgcnn_cls'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training [default: 32]')
    parser.add_argument('--epoch', default=250, type=int, help='Number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate [default: 1e-4]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Pptimizer for training [default: SGD]')
    parser.add_argument('--gpu', type=str, default='0', help='Specify gpu device [default: 0]')
    parser.add_argument('--local_rank', type=int, default=-1, help='Specify  local rank [default: -1]')
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


def test(model, infer, loader):
    mean_gen_type_correct = []
    mean_target_type_correct = []
    for j, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        _, points, target = data

        trot = None
        if args.rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points)
        if torch.cuda.is_available():
            if DEBUG:
                points, target = points.cuda(), target.cuda()
            else:
                points, target = points.cuda(args.local_rank), target.cuda(args.local_rank)
        center_coords, coords, types = model(points)
        loss, gen_type_loss, target_type_loss, emd_loss, gen_type_correct, target_type_correct = infer(
            center_coords, coords, types, target)
        mean_gen_type_correct.append(gen_type_correct.cpu())
        mean_target_type_correct.append(target_type_correct.cpu())
    instance_gen_type_acc = np.mean(mean_gen_type_correct)
    instance_target_type_acc = np.mean(mean_target_type_correct)
    return instance_gen_type_acc, instance_target_type_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not DEBUG:
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        torch.cuda.set_device(args.local_rank)  # 作用相当于CUDA_VISIBLE_DEVICES命令，修改环境变量
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
    if DEBUG:
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, collate_fn=clfn.collate_ignore_none,
                                                      batch_size=args.batch_size,
                                                      shuffle=True, num_workers=1)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(TRAIN_DATASET)
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, collate_fn = clfn.collate_ignore_none,
                                                      batch_size=args.batch_size,
                                                      sampler=train_sampler, num_workers=4)
    validDataLoader = torch.utils.data.DataLoader(VALID_DATASET, collate_fn = clfn.collate_ignore_none, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if DEBUG:
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, collate_fn=clfn.collate_ignore_none,
                                                     batch_size=args.batch_size,
                                                     shuffle=False, num_workers=1)
    else:
        test_sampler = torch.utils.data.distributed.DistributedSampler(TEST_DATASET, shuffle=False)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, collate_fn = clfn.collate_ignore_none,
                                                     batch_size=args.batch_size,
                                                     sampler=test_sampler, num_workers=4)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(last_npoint=10, atom_num_per_last_point = 10, atom_type_num = 10, normal_channel=args.normal)
    criterion = MODEL.get_loss()
    # model_stat.getModelSize(classifier)
    if torch.cuda.is_available():
        if DEBUG:
            criterion.cuda()
        else:
            classifier.to(args.local_rank)
            criterion.to(args.local_rank)
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        if DEBUG:
            classifier = classifier.cuda()
        else:
            classifier = DDP(classifier, device_ids=[args.local_rank], output_device=args.local_rank)
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
    best_instance_acc1 = 0.0
    best_instance_acc2 = 0.0
    best_epoch1 = 0
    best_epoch2 = 0
    mean_gen_type_correct = []
    mean_target_type_correct = []
    '''TRANING'''
    logger.info('Start training...')
    with torch.no_grad():
        instance_acc1, instance_acc2 = test(classifier, criterion, testDataLoader)
        log_string('Test Instance Accuracy gen->target:  %.4f, Instance Accuracy target->gen:       %.4f' % (
        instance_acc1, instance_acc2))
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        if not DEBUG:
            trainDataLoader.sampler.set_epoch(epoch)
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
            #todo drop rate is too high
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)

            if torch.cuda.is_available():
                if DEBUG:
                    points, target = points.cuda(), target.cuda()
                else:
                    points, target = points.cuda(args.local_rank), target.cuda(args.local_rank)
            optimizer.zero_grad()
            center_coords, coords, types = classifier(points)
            loss, gen_type_loss, target_type_loss, emd_loss, gen_type_correct,  target_type_correct = criterion(center_coords, coords, types, target)
            mean_gen_type_correct.append(gen_type_correct.cpu())
            mean_target_type_correct.append(target_type_correct.cpu())
            loss.sum().backward()
            optimizer.step()
            global_step += 1

        log_string('Train Instance gen type Accuracy: %f' % np.mean(mean_gen_type_correct))
        log_string('Train Instance target type Accuracy: %f' % np.mean(mean_target_type_correct))
        scheduler.step()
        with torch.no_grad():
            instance_acc1, instance_acc2 = test(classifier, criterion, testDataLoader)
            if (instance_acc1 >= best_instance_acc1):
                best_instance_acc1 = instance_acc1
                best_epoch1 = epoch + 1

            if (instance_acc2 >= best_instance_acc2):
                best_instance_acc2 = instance_acc2
                best_epoch2 = epoch + 1
            log_string('Test Instance Accuracy gen->target:  %.4f, Instance Accuracy target->gen:       %.4f' % (instance_acc1, instance_acc2))
            log_string('Best Instance Accuracy gen->target:  %.4f, Best Instance Accuracy gen->target:  %.4f' % (best_instance_acc1, best_instance_acc2))

            if (instance_acc1 >= best_instance_acc1 and instance_acc2 >= best_instance_acc2):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch1,
                    'instance_acc gen2target': instance_acc1,
                    'instance_acc target2gen': instance_acc2,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
