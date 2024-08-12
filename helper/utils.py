import torch
import numpy as np
import cv2

def _seed_everything(random_seed=41):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def accuracy(output, target, topk=[1]):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def prob_loss(logits, target):
    assert len(target) == len(logits)
 
    log_probs = logits.log_softmax(dim=-1)
    return -(log_probs * target).sum(dim=-1).mean()


def save_model(save_dir,
               epoch,
               model,
               optimizer,
               lr_scheduler,
               device_ids,
               best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str('_'.join([save_dir, 'current.pth']))
    torch.save(state, filename)

    if best:
        filename = str('_'.join([save_dir, 'best.pth']))
        torch.save(state, filename)


def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
            n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def count_parameters(model):
    "Couting number of trainable params"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def L2Norm(model):
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    return l2_reg


def set_trainable(model, boolean: bool = True, except_layers: list = [], device_ids: list = []):
    if boolean:
        for name, param in model.named_parameters():
            if name in except_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        #         assert len(except_layers) > 0, "Require free layer"
        for name, param in model.named_parameters():
            if name in except_layers:
                param.requires_grad = True
                print('--------',name)
            else:
                param.requires_grad = False

    print("Training params: ", count_parameters(model))
    return model


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.TRAIN.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.TRAIN.lr * (opt.TRAIN.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            if isinstance(num, str):
                self.file.write(num)
            else:
                self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
