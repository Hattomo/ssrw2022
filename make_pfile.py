import os
import platform
import json
from logging import getLogger, config
from datetime import datetime
import subprocess

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp

# 自作プログラムの読込
from my_args import get_parser, set_debug_mode, set_release_mode
from my_utils import my_util
from my_dataset_av import MyCollator, ROHANDataset
from model import CNN3LSTMCTC, CNN2LSTMCTC, CNNConformer
import makelabel
import data_transform


# format outputs
def format_outputs(verbose):
    predict = [verbose[0]]
    for i in range(1, len(verbose)):
        if verbose[i] != predict[-1]:
            predict.append(verbose[i])
    predict = [l for l in predict if l != phones.index('sil')]
    return predict


def predict_test(model, loader):
    model.eval()
    data_num = len(loader.dataset)
    with torch.no_grad():
        for i, (inputs, labels, input_lengths, target_lengths, dlib) in enumerate(loader):
            input_lengths = input_lengths.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)
            labels = labels.to(DEVICE)
            if opts.mode == "FV":
                inputs = inputs.to(DEVICE).float()
                dlib = dlib.to(DEVICE).float()
                outputs = model(inputs, dlib)
            # 結果保存用
            batch_size = inputs.size(0)
            # input_lengths = torch.full((1, batch_size), fill_value=outputs.size(0))
            result_text = ""

            for i in range(batch_size):
                output = outputs[i]
                label = labels[i]
                _, output = output.max(dim=1)
                pred = format_outputs(output)
                output = [phones[l] for l in output]
                pred = [phones[l] for l in pred]
                label = [phones[l] for l in label if phones[l] not in "sil"]
                ter = my_util.calculate_error(pred, label)
                result_text += "-"*50 + "\n\n---Output---\n" + " ".join(output) + "\n\n---Predict---\n" + " ".join(
                    pred) + "\n\n---Label---\n" + " ".join(label) + "\n\n" + "WER : " + "\n\n"
            outputs = outputs.permute(1, 0, 2).log_softmax(2)
            ctc_loss = criterion(outputs, labels, input_lengths, target_lengths)
    with open("build/result.txt", mode='w') as f:
        f.write(result_text)

if __name__ == "__main__":

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    # Load parameters
    parser = get_parser(start_time)
    opts = parser.parse_args()

    # Set Device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opts.device = DEVICE

    phones = makelabel.get_phones_csv(opts.label)

    dict = {phones[i]: i for i in range(len(phones))}

    label = makelabel.get_label_csv(opts.label, phones)
    transform = data_transform.DataTransform()
    testset = ROHANDataset(labels=label[opts.test_size[0]:opts.test_size[1]],
                        image_path=opts.image_path,
                        csv_path=opts.csv_path,
                        data_size=opts.test_size,
                        opts=opts,
                        transform=transform.base_img_transform)
    collate_fn = MyCollator(phones)
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=opts.batch_size,
                                            shuffle=False,
                                            num_workers=opts.workers,
                                            collate_fn=collate_fn,
                                            pin_memory=False)

    # Set model
    output_size = len(phones)
    a = torch.Tensor([0])
    model = CNN2LSTMCTC(opts.lstm_hidden, output_size, opts).to(DEVICE)
    print('model build')
    #- Set Loss func
    criterion = nn.CTCLoss(blank=phones.index('sil'), zero_infinity=True)
    # 保存したものがあれば呼び出す
    if opts.resume:
        if os.path.isfile(opts.resume):
            print("=> loading checkpoint '{}'".format(opts.resume))
            checkpoint = torch.load(opts.resume)
            START_EPOCH = checkpoint['epoch']
            best_val = float('inf')
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            best_val = float('inf')
    else:
        best_val = float('inf')

    predict_test(model, testloader)
