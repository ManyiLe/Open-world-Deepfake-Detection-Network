import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device_ids=[0, 1]

import warnings
import random 
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter
from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.ODDN import ODDN
from networks.pcgrad import PCGrad
from options.train_options import TrainOptions
from helper.utils import set_trainable
from eval_test_mygen9GANs import test_mygen9GANs
from eval_test8gan import test_8GANs

warnings.filterwarnings('ignore')


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)

    val_opt.dataroot = '/opt/data/private/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/'
    val_opt.classes = ['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
                    'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

    val_opt.dataroot = os.path.join(val_opt.dataroot, val_opt.val_split) 
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = True
    val_opt.serial_batches = True
    val_opt.jpg_method = ['cv2'] 
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    torch.cuda.empty_cache()

    seed_everything(200371)
    opt = TrainOptions().parse()
    val_opt = get_val_opt()

    #-----customize Hyper Parameter, you can also set them in commander line parameter--------#
    opt.dataroot = '/opt/data/private/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/'
    opt.batch_size = 128
    opt.num_threads = 16
    opt.classes = ['chair', 'horse']
    opt.earlystop_epoch = 3
    opt.blur_prob = 0.5 
    opt.blur_sig = [0,3] # blur sample range
    opt.jpg_prob = 0.2
    #opt.jpg_qual = range(30,101) in dataset.py, we set compression val as 60(C40) when training
    opt.warming_up = 3 
    opt.adv_warmup = 10
    opt.lr = 0.0002
    max_grad_norm = 1
    #-------------------------------------------------------------------------------------------#

    opt.dataroot = os.path.join(opt.dataroot, opt.train_split)
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = ODDN(opt).to('cuda')
    hard_share_param = ["resnet.module.fc.weight","resnet.module.fc.bias"]
    print("#----Warm up Parameter----#")
    for name, p in model.named_parameters():
        if "resnet" not in name:
            hard_share_param.append(name)
    set_trainable(model, False, hard_share_param, [])
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.module

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr, betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-4,amsgrad=False)
    else:
        raise ValueError("optim should be [adam, sgd]")
    optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids).module
    pc_adam = PCGrad(opt, model, optimizer, max_grad_norm)

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.0001, verbose=True)

    for epoch in range(opt.niter):

        if opt.warming_up == epoch:
                set_trainable(model, True, [], device_ids)

        for data in tqdm(data_loader):

            if epoch >= opt.adv_warmup:
                alpha = opt.alpha
            else:
                alpha = None
            
            model.train()
            model.total_steps += 1

            #0 ture or not compressed / 1 fake or compressed
            aug_input, no_aug_input, tf_label, cmp_label = data
            cmp_label_np = np.array(cmp_label)
            input = torch.cat([no_aug_input[cmp_label_np], aug_input[cmp_label_np], aug_input[~cmp_label_np]], dim=0)
            mask_label = np.ones(len(input)).astype(bool)
            mask_label[sum(cmp_label_np)*2:] = False
            tf_label_np = np.array(torch.cat([tf_label[cmp_label], tf_label[cmp_label], tf_label[~cmp_label]], dim=0)).astype(bool)
            label = [tf_label_np, mask_label]
            input = input.cuda()

            tf_loss, cmp_loss, dis_loss, _ = model(input, True, label, alpha)

            loss = tf_loss
            if dis_loss is not None:
                loss += dis_loss

            pc_adam.zero_grad()            
            if epoch >= opt.adv_warmup and cmp_loss is not None:
                pc_adam.set_loss(cmp_loss)
                pc_adam.pc_backward(epoch, [loss, cmp_loss])#gradient alignment
            else:
                pc_adam.pc_backward(epoch, [loss])
            pc_adam.step()
            
            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} cmp_loss: {},at step: {}".format(loss, cmp_loss ,model.total_steps))
                train_writer.add_scalar('loss', loss, model.total_steps)
            torch.cuda.empty_cache()

        # Validation
        model.eval()
        acc, ap = validate(model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        save_filename = 'model_epoch_%s.pth' % epoch
        state_dict = {'model': model.state_dict()}
        save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name), save_filename)
        torch.save(state_dict, save_path)

        #test9GANs
        accs, aps, avg_acc, avg_ap = test_mygen9GANs(model, False)
        for key, value in accs.items():
            val_writer.add_scalar(key + "acc", value, epoch)
        for key, value in aps.items():
            val_writer.add_scalar(key + "ap", value, epoch)
        val_writer.add_scalar("avg_acc_9GANs", avg_acc, epoch)
        val_writer.add_scalar("avg_ap_9GANs", avg_ap, epoch)
        print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format("test9GAN",'Mean', avg_acc, avg_ap))

        accs, aps, avg_acc, avg_ap = test_mygen9GANs(model, True)
        for key, value in accs.items():
            val_writer.add_scalar(key + "acc" + "cmp50", value, epoch)
        for key, value in aps.items():
            val_writer.add_scalar(key + "ap" + "cmp50", value, epoch)
        val_writer.add_scalar("avg_acc_9GANs" + "cmp50", avg_acc, epoch)
        val_writer.add_scalar("avg_ap_9GANs" + "cmp50", avg_ap, epoch)
        print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format("test9GAN CMP",'Mean', avg_acc, avg_ap))

        #test8GANs
        accs, aps, avg_acc, avg_ap = test_8GANs(model, False)
        for key, value in accs.items():
            val_writer.add_scalar(key + "acc", value, epoch)
        for key, value in aps.items():
            val_writer.add_scalar(key + "ap", value, epoch)
        val_writer.add_scalar("avg_acc_8GANs", avg_acc, epoch)
        val_writer.add_scalar("avg_ap_8GANs", avg_ap, epoch)
        print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format("test8GAN",'Mean', avg_acc, avg_ap))

        accs, aps, avg_acc, avg_ap = test_8GANs(model, True)
        for key, value in accs.items():
            val_writer.add_scalar(key + "acc" + "cmp50", value, epoch)
        for key, value in aps.items():
            val_writer.add_scalar(key + "ap" + "cmp50", value, epoch)
        val_writer.add_scalar("avg_acc_8GANs" + "cmp50", avg_acc, epoch)
        val_writer.add_scalar("avg_ap_8GANs" + "cmp50", avg_ap, epoch)
        print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format("test8GAN CMP",'Mean', avg_acc, avg_ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate(optimizer)
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.0001, verbose=True)
            else:
                print("Early stopping.")
                break