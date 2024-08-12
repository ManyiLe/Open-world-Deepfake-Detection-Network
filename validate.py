import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
from data.datasets import*


def validate(model, opt):
    
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for data in tqdm(data_loader):

            input, _, tf_label, cmp_label = data

            input = input.cuda()
            tf_label = tf_label.cuda().float()

            tf_output = model(input)[-1]
            
            y_pred.extend(tf_output.sigmoid().flatten().tolist())
            y_true.extend(tf_label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    try:
        fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred, pos_label=1)
    except:
        # for the case when we only have one sample
        return acc, ap, 0, None, r_acc, f_acc, y_true, y_pred

    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        # for the case when all the samples within a batch is fake/real
        auc, eer = 0, None
    else:
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return acc, ap, auc, eer, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    """
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
    """
