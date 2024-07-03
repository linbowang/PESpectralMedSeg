import argparse
import logging
import os
import random
import sys

import numpy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from PvtMissFormer_lzh.PvtFormer_Spe_35 import PvtMissFormer
from loguru import logger

seed = 1234
random.seed(seed)
numpy.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str,
                    default='PvtMissFormer_lzh', help='the name of network')
parser.add_argument('--volume_path', type=str,
                    default='./data/Synapse/', help='root dir for validation volume data')  # for Synapse volume_path=root_dir
parser.add_argument('--data_name', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--train_out_dir', type=str,
                    default='./train_out_224', help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
# parser.add_argument('--test_save_dir', type=str, default='./train_out_PVT_Unet/predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.05, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.data_name == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return performance, mean_hd95


if __name__ == "__main__":
    net = PvtMissFormer(num_classes=args.num_classes).cuda(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }

    dataset_name = args.data_name
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # 配置log
    log_path = './test/log/{}/{}_log.log'.format(args.method, args.data_name)
    logger.add(log_path)

    best_dice = 0
    best_epoch = 0
    best_hd95 = 99
    for i in range(0, 199, 1):
        model_path = '{}/{}/models/{}_model/'.format(args.train_out_dir, args.method, args.data_name)
        # model_path = '{}/{}/{}_model/'.format(args.train_out_dir, args.method, args.data_name)
        model_name = 'epoch_' + str(i) + '.pth'
        snapshot = os.path.join(model_path, model_name)
        msg = net.load_state_dict(torch.load(snapshot))
        logger.info(msg)
        snapshot_name = snapshot.split('/')[-1]
        logger.info('---------' + args.method + '   ' + snapshot_name + '---------')

        args.is_savenii = False
        if args.is_savenii:
            args.test_save_dir = os.path.join(args.train_out_dir, "predictions")
            test_save_path = args.test_save_dir + '/{}/{}/epoch_'.format(args.method, args.data_name) + str(i)
            os.makedirs(test_save_path, exist_ok=True)
        else:
            test_save_path = None

        dice_i, hd95_i = inference(args, net, test_save_path)
        logger.info("{}  epoch_{}   dice={}   hd_95={}".format(args.method, i, dice_i, hd95_i))
        if best_dice < dice_i:
            best_dice = dice_i
            best_epoch = i
            best_hd95 = hd95_i
            logger.info(
                '********************  best_epoch={}, best_dice={} , best_hd95={}  ********************'.format(
                    best_epoch, best_dice, best_hd95))
    logger.info(
            '********************  best_epoch={}, best_dice={} , best_hd95={}  ********************'.format(
                best_epoch, best_dice, best_hd95))


