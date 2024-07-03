import argparse
import os
import random
import numpy
import cv2
import torch
from torch import optim
from torch.backends import cudnn
from tqdm import tqdm
from loguru import logger

from PvtMissFormer_lzh.PvtFormer_Spe_35 import PvtMissFormer
from utils import ensure_dir, DiceLoss
from torchvision import transforms

from datasets.dataset_synapse import Synapse_dataset
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

# 随机数
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
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--save_epoch', type=int,
                    default=1, help='save per epoch')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/', help='root dir for data')
parser.add_argument('--data_name', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--channel_n', type=int,
                    default=1, help='channel of data')
# parser.add_argument('--output_dir', type=str,
#                     default='./train_out_PVT_Unet', help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=90000, help='maximum epoch number to train')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
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
    args.root_path = os.path.join(args.root_path, "train_npz")


def single_gpu_train(args):

    save_epoch = args.save_epoch
    model_save_path = 'train_out_224/{}/models/{}_model'.format(args.method, args.data_name)
    ensure_dir(model_save_path)

    net = PvtMissFormer(num_classes=args.num_classes).cuda(0)
    net.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = optim.AdamW(net.parameters(), lr=args.base_lr, weight_decay=0.0001)

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir,
                            split="train", img_size=args.img_size,
                            norm_x_transform=x_transforms, norm_y_transform=y_transforms)

    logger.info("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logger.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        epoch_loss = 0
        net.train()
        # ####################################### Train ########################################
        for i, data in enumerate(train_loader):
            inputs, labels = data['image'], data['label']
            inputs = inputs.cuda()  # b 1 224 224
            labels = labels.squeeze(1).cuda()  # b 1 224 224

            outputs = net(inputs)
            loss_ce = ce_loss(outputs, labels[:].long())
            loss_dice = dice_loss(outputs, labels, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            total_loss = loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            epoch_loss += total_loss.item()

            logger.info('epoch: %d iteration %d : loss : %f, loss_ce: %f' % (epoch, iter_num, loss.item(), loss_ce.item()))
        logger.info("")
        logger.info('{}     epoch_{}   epoch_loss:{}'.format(args.method, epoch, epoch_loss))

        if epoch % save_epoch == 0:
            torch.save(net.state_dict(), model_save_path + '/epoch_' + str(epoch) + '.pth')
            logger.info("epoch_{} save successfully".format(epoch))


if __name__ == '__main__':

    log_path = './train_out_224/{}/log/{}_log.log'.format(args.method, args.data_name)
    logger.add(log_path)

    single_gpu_train(args)


