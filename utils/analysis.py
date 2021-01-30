import numpy as np
import torch
import os.path as osp
import glob
import matplotlib.pyplot as plt

cnames = [
    'green',
    'red',
    'blue',
    'brown',
    'black',
    'teal',
    'forestgreen',
    'maroon',
    'midnightblue',
]
lnames = [
    '-',
    '--',
    '-.',
    ':',
    '-',
    '--',
    '-.',
    ':',
    '-',
    '--',
    '-.',
    ':',
]

def draw_line(log_dict, offset=0):
    handle_list, label_list = [], []
    for idx, (key, val) in enumerate(log_dict.items()):
        hd, = plt.plot(val[0][offset:], val[1][offset:], color=cnames[idx], linewidth=1.0, linestyle=lnames[idx])
        handle_list.append(hd)
        label_list.append(key)
    plt.legend(handles=handle_list, labels=label_list, loc='upper left')
    plt.show()


if __name__ == '__main__':
    model_dir = '/home/songkey/workspace/log2021/0104/exp_log'
    model_list = sorted(glob.glob(osp.join(model_dir, "*.pth")))

    log_dict = dict()
    for model_path in model_list:
        base_name, _ = osp.splitext(osp.basename(model_path))
        key = '_'.join(base_name.split('_')[2:4] + [base_name.split('_')[-1]])

        test_hist = torch.load(model_path, map_location={'cuda': 'cpu'})['visdom_data']['test_list_hist']
        epoch_list, loss_list = [], []
        for line in test_hist:
            epoch = int(line.split()[2].replace('epoch', ''))
            loss = float(line.split(':')[-1])
            epoch_list.append(epoch)
            loss_list.append(loss)



        log_dict[key] = (np.array(epoch_list, dtype=np.int), np.array(loss_list, dtype=np.float32))

    draw_line(log_dict, offset=0)