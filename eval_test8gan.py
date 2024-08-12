import time
import os
import torch

from util import Logger
from validate import validate
from options.test_options import TestOptions
from networks.ODDN import ODDN
import numpy as np
import warnings
warnings.filterwarnings("ignore")

vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

def test_8GANs(model, cmp = False):

    opt = TestOptions().parse(print_options=False)
    opt.dataroot = '/opt/data/private/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
    dataroot = opt.dataroot
    opt.batch_size = 32
    if cmp:
        opt.cmp = True
        opt.cmp_arg = 60 #quality aware

        #opt.jpg_prob = 1 quality agnostic
        #opt.jpg_qual = range(30,101)
    else:
        opt.cmp = False
    
    accs = {}; aps = {}
    model.eval()
    with torch.no_grad():
        for v_id, val in enumerate(vals):
            opt.dataroot = '{}/{}'.format(dataroot, val)
            opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
            opt.no_resize = False    # testing without resizing by default
            opt.no_crop = True    # testing without resizing by default

            acc, ap, auc, _, _, _, _, _ = validate(model, opt)
            accs[val] = acc * 100; aps[val] = ap * 100
    
    avg_acc , avg_ap = [ value  for value in accs.values()],  [ value  for value in aps.values()]
    avg_acc , avg_ap = sum(avg_acc) / len(avg_acc), sum(avg_ap) / len(avg_ap)

    return accs, aps, avg_acc, avg_ap


if __name__ == "__main__":
    """
    model_path = './checkpoints/resnet-2class-iccv+att-0.1cmp-0.5blur-0.0001lr2024_07_17_17_16_34/model_epoch_best.pth'

    results_dir = './results_onprogan/'
    logpath = os.path.join(results_dir, model_path.split('/')[-2])
    os.makedirs(results_dir, mode = 0o777, exist_ok = True) 
    os.makedirs(logpath, mode = 0o777, exist_ok = True) 
    Logger(os.path.join(logpath, model_path.split('/')[-1] + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+'.log'))


    dataroot = '/opt/data/private/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
    print(f'Dataroot {dataroot}')
    print(f'Model_path {model_path}')

    accs = [];aps = [];aucs=[]
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    opt = TestOptions().parse(print_options=False)
    model = QDNetwork(opt)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to('cuda')
    model.eval()

    accs, aps, avg_acc, avg_ap = test_8GANs(model, False)

    for v_id, val in enumerate(vals):
        print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id, val, accs[val], aps[val]))
    print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id+1,'Mean', avg_acc, avg_ap))
    print('*'*25)
    """
