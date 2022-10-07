# -*- coding: utf-8 -*-

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
from torch.utils.tensorboard import SummaryWriter  # tensorboard
from tqdm import tqdm
from torchinfo import summary
from full_model import SSRWFullModel

# 自作プログラムの読込
from my_args import get_parser, set_debug_mode, set_release_mode
from my_utils import my_util
from my_dataset_av import MyCollator, ROHANDataset
from model import CNN3LSTMCTC, CNN2LSTMCTC, CNNConformer
import makelabel
import data_transform
from torchinfo import summary
from language_model import PhonemeLangModel
import json
from pad_label_func import pad_label
from pad_out_func import pad_out

dict_path = 'data/label.pkl'
labels_path = 'data/phoneme.csv'
dict_path = 'data/phoneme_dict.json'
with open(dict_path, 'r') as f:
    phone_dict = json.load(f)

torch.multiprocessing.set_start_method('spawn')

start_time = datetime.now().strftime("%y%m%d-%H%M%S")

# Load parameters
parser = get_parser(start_time)
opts = parser.parse_args()

# Initalize logger
with open(opts.logger_config_path, 'r') as f:
    log_conf = json.load(f)

# set debug mode
if opts.debug:
    set_debug_mode(opts, log_conf)
if not opts.debug:
    set_release_mode(opts, log_conf)

config.dictConfig(log_conf)
progress_logger = getLogger("Progress")
result_logger = getLogger("Result")

progress_logger.info(f"Start Time: {start_time}")

# set random seed
torch.manual_seed(opts.seed)
progress_logger.info(f"opts -> {opts}")

# yapf: disable
progress_logger.info(f"Machine    : {platform.machine()},\n\
Processor  : {platform.processor()},\n\
Platform   : {platform.platform()},\n\
Machine    : {platform.machine()},\n\
python     : {platform.python_version()},\n\
Pytorch    : {torch.__version__}"                                 )
# yapf: enable

DEVICE = torch.device('cuda')

# save args
with open(f"{opts.base_path}/args.json", 'wt') as f:
    json.dump(vars(opts), f, indent=4)

branch = subprocess.run(["git", "branch", "--contains=HEAD"], encoding='utf-8', stdout=subprocess.PIPE)

commit_hash = subprocess.run(["git", "rev-parse", "HEAD"], encoding='utf-8', stdout=subprocess.PIPE)

with open(f"{opts.base_path}/environment.yml", mode='w', encoding="utf-8") as f:
    conda_env = subprocess.run(["conda", "env", "export"], encoding='utf-8', stdout=subprocess.PIPE)
    f.write(conda_env.stdout)

progress_logger.info(f"branch : " + branch.stdout)
progress_logger.info(f"commit hash : " + commit_hash.stdout)

with open(f"{opts.base_path}/diff.patch", mode='w', encoding="utf-8") as f:
    diff_patch = subprocess.run(["git", "diff", "--diff-filter=d", "./"], encoding='utf-8', stdout=subprocess.PIPE)
    f.write(diff_patch.stdout)

# Init tensorboard
if not opts.no_check:
    my_util.check_file(opts.tensorboard)
writer = SummaryWriter(log_dir=opts.tensorboard)

progress_logger.info(f"Set tensorboard file : {opts.tensorboard}")

# Init checkpoint
if not opts.no_check:
    my_util.check_file(opts.checkpoint)
os.makedirs(opts.checkpoint, exist_ok=True)
progress_logger.info(f"Set checkpoint path : {opts.checkpoint}")

opts.device = DEVICE

progress_logger.info(f"DEVICE : {DEVICE}")
# cudnnの自動チューナー:場合によって速くなったり遅くなったり(入力サイズが常に一定だといいらしいが)
cudnn.benchmark = opts.benchmark  #fででもいい
progress_logger.info(f"cudnn.benchmark : {opts.benchmark}")
progress_logger.info("Initilaize complete🎉")

progress_logger.info("Start defining data loader...")

# phones = makelabel.get_phones_csv(opts.label, opts.train_size[0], opts.test_size[1])
phones = makelabel.load_phones_csv()

dict = {phones[i]: i for i in range(len(phones))}
with open(opts.token, 'w') as vocab_file:
    json.dump(dict, vocab_file, indent=4, ensure_ascii=False)

label = makelabel.get_label_csv(opts.label, phones)
transform = data_transform.DataTransform()
# Load dataset
progress_logger.info("Loading Dataset")
trainset = ROHANDataset(labels=label[opts.train_size[0]:opts.train_size[1]],
                        image_path=opts.image_path,
                        csv_path=opts.csv_path,
                        data_size=opts.train_size,
                        opts=opts,
                        transform=transform.train_img_transform)
validset = ROHANDataset(labels=label[opts.valid_size[0]:opts.valid_size[1]],
                        image_path=opts.image_path,
                        csv_path=opts.csv_path,
                        data_size=opts.valid_size,
                        opts=opts,
                        transform=transform.base_img_transform)
testset = ROHANDataset(labels=label[0:20],
                       image_path=opts.image_path,
                       csv_path=opts.csv_path,
                       data_size=opts.test_size,
                       opts=opts,
                       transform=transform.base_img_transform)
collate_fn = MyCollator(phones)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=opts.batch_size,
                                          shuffle=True,
                                          num_workers=opts.workers,
                                          collate_fn=collate_fn,
                                          pin_memory=False)
validloader = torch.utils.data.DataLoader(validset,
                                          batch_size=opts.batch_size,
                                          shuffle=False,
                                          num_workers=opts.workers,
                                          collate_fn=collate_fn,
                                          pin_memory=False)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=opts.batch_size,
                                         shuffle=False,
                                         num_workers=opts.workers,
                                         collate_fn=collate_fn,
                                         pin_memory=False)

progress_logger.info("Data load complete")

progress_logger.info("Defining model...")

# Set model
output_size = len(phones)
ctc_model = CNNConformer(opts.lstm_hidden, output_size, opts).to(DEVICE)
progress_logger.info(f"Model : {ctc_model}")
# TODO: Use torch summary (info?)
# progress_logger.info(summary(model, [(6, 237, 1, 128, 128), (6, 237, 136)],device=opts.device))

for param in ctc_model.efficient_net.parameters():
    param.requires_grad = False
for param in ctc_model.efficient_net._fc.parameters():
    param.requires_grad = True
for param in ctc_model.efficient_net._bn1.parameters():
    param.requires_grad = True
for param in ctc_model.efficient_net._conv_head.parameters():
    param.requires_grad = True
for param in ctc_model.efficient_net._blocks[-1].parameters():
    param.requires_grad = True
for param in ctc_model.efficient_net._blocks[-2].parameters():
    param.requires_grad = True

#- Set Loss func
criterion = nn.CrossEntropyLoss()

# 保存したものがあれば呼び出す
ctc_model.load_state_dict(torch.load('./checkpoints_aoyama/epoch029_val1.192.pth', map_location="cuda:0")['model_state_dict'])
language_model = PhonemeLangModel(device=DEVICE).to(DEVICE)
language_model.load_state_dict(torch.load('./checkpoints_aoyama/lstm_lang_model_ROHAN.pth'))
model = SSRWFullModel(ctc_model, language_model, phoneme_dict=phone_dict, device=DEVICE).to(DEVICE)

# optimizer定義 adamかsgd?
optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, eps=1e-03, weight_decay=0)  #lr(学習率)はいじった方がいい0.001

progress_logger.info("Model defined")

def detach(states):
    return [state.detach() for state in states]

# format outputs
def format_outputs(verbose):
    predict = [verbose[0]]
    for i in range(1, len(verbose)):
        if verbose[i] != predict[-1]:
            predict.append(verbose[i])
    predict = [l for l in predict if l != phones.index('_')]
    return predict

def train(loader, model, criterion, optimizer, scaler, epoch):
    """

    1 epoch on the training set

    Args:
        loader : pytorch loader
        model : pytorch model
        criterion : pytorch loss function
        optimizer : pytorch optimizer
        epoch (int): epoch number

    Returns:
        ctc_losses.avg (float) : avarage criterion loss

    """
    progress_logger.info(f"Training phase , epoch = {epoch}")
    data_manager = my_util.DataManager("train", writer)
    data_num = len(loader.dataset)  # テストデータの総数
    pbar = tqdm(total=int(data_num / opts.batch_size))
    hidden_dim = model.language_model.hidden_dim_encoder
    batch_size = opts.batch_size
    states = (torch.zeros(1, batch_size, hidden_dim).to(DEVICE),
              torch.zeros(1, batch_size, hidden_dim).to(DEVICE))
    model.train()
    for batch, (inputs, targets, input_lengths, target_lengths, dlib) in enumerate(loader):
        # データをdeviceに載せる
        # 初期値
        input_lengths = input_lengths.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)
        targets = targets.to(DEVICE, non_blocking=True)
        inputs = inputs.to(DEVICE, non_blocking=True).float()
        dlib = dlib.to(DEVICE, non_blocking=True).float()
        # 統合モデルへの入力
        outputs, states = model(inputs, dlib, input_lengths, False, states)
        batch_size = inputs.size(0)
        # 出力とラベルのpadding
        if targets.shape[1] < outputs.shape[1]:
            targets = pad_label(targets, outputs).to(DEVICE)
        elif targets.shape[1] > outputs.shape[1]:
            outputs = pad_out(targets, outputs).to(DEVICE)
        # nn.CrossEntropyLossに入れれるようにreshape
        outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2])
        targets = targets.reshape(targets.shape[0]*targets.shape[1])
        loss = criterion(output, targets)
        data_manager.update_loss(loss.data.item(), batch_size)
        result_text = ""
        for i in range(batch_size):
            output = outputs[i]
            label = targets[i]
            _, output = output.max(dim=1)
            pred = format_outputs(output)
            output = [phones[l] for l in output]
            pred = [phones[l] for l in pred]
            label = [phones[l] for l in label if phones[l] not in "_"]
            ter = my_util.calculate_error(pred, label)
            data_manager.update_acc(ter, batch_size)
            result_text += "-"*50 + "\n\n---Output---\n" + " ".join(output) + "\n\n---Predict---\n" + " ".join(
                pred) + "\n\n---Label---\n" + " ".join(label) + "\n\n"
        result_logger.info(result_text)
        loss.backward()  # calculate gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip)  # clip gradients
        optimizer.step()
        optimizer.zero_grad()  # initlaize grad
        pbar.update(1)
    data_manager.write(epoch)
    with open("build/result.txt", mode='w') as f:
        f.write(result_text)
    pbar.close()
    progress_logger.info('Epoch: {0}\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, loss=data_manager.loss_manager.loss))

def valid(loader, model, criterion, epoch):
    """

    valid function for 1 epoch
        valid train
        write result data

    Args:
        loader : pytorch dataloader
        model : pytorch model
        criterion : pytorch loss function
        epoch (int) : epoch number

    Returns:
        ctc_losses.avg (float?) : avarage criterion loss
        accs.avg (float?) : avarage accuracy

    """
    progress_logger.info("Validation phase")
    # 各値初期化
    data_manager = my_util.DataManager("valid", writer)
    model.eval()
    data_num = len(loader.dataset)  # テストデータの総数
    pbar = tqdm(total=int(data_num / opts.batch_size))
    batch_size = opts.batch_size
    hidden_dim = model.language_model.hidden_dim_encoder
    device = model.laguage_model.device
    states = (torch.zeros(1, batch_size, hidden_dim).to(device),
              torch.zeros(1, batch_size, hidden_dim).to(device))
    with torch.no_grad():
        for i, (inputs, labels, input_lengths, target_lengths, dlib) in enumerate(loader):
            # データをdeviceに載せる
            # 初期値
            input_lengths = input_lengths.to(DEVICE, non_blocking=True)
            target_lengths = target_lengths.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            inputs = inputs.to(DEVICE, non_blocking=True).float()
            dlib = dlib.to(DEVICE, non_blocking=True).float()
            outputs = model(inputs, dlib, input_lengths, False, states)
            # 結果保存用
            batch_size = inputs.size(0)
            # 統合モデルへの入力
            outputs = model(inputs, dlib, input_lengths, False, states)
            batch_size = inputs.size(0)
            # 出力とラベルのpadding
            if targets.shape[1] < outputs.shape[1]:
                targets = pad_label(targets, outputs).to(DEVICE)
            elif targets.shape[1] > outputs.shape[1]:
                outputs = pad_out(targets, outputs).to(DEVICE)
            # nn.CrossEntropyLossに入れれるようにreshape
            outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2])
            targets = targets.reshape(targets.shape[0]*targets.shape[1])
            loss = criterion(output, targets)
            data_manager.update_loss(loss.data.item(), batch_size)
            # input_lengths = torch.full((1, batch_size), fill_value=outputs.size(0))
            result_text = ""
            for i in range(batch_size):
                output = outputs[i]
                label = labels[i]
                _, output = output.max(dim=1)
                pred = format_outputs(output)
                output = [phones[l] for l in output]
                pred = [phones[l] for l in pred]
                label = [phones[l] for l in label if phones[l] not in "_"]
                ter = my_util.calculate_error(pred, label)
                data_manager.update_acc(ter, batch_size)
                result_text += "-"*50 + "\n\n---Output---\n" + " ".join(output) + "\n\n---Predict---\n" + " ".join(
                    pred) + "\n\n---Label---\n" + " ".join(label) + "\n\n"
            result_logger.info(result_text)
            # measure performance and record loss
            data_manager.update_loss(loss.data.item(), batch_size)
            pbar.update(1)
    pbar.close()
    data_manager.write(epoch)
    with open("build/result.txt", mode='w') as f:
        f.write(result_text)
    progress_logger.info('Val loss:{loss.avg:.4f} '
                         'Acc:{Acc.avg:4f}'.format(loss=data_manager.loss_manager.loss,
                                                   Acc=data_manager.acc_manager.total_error))
    return data_manager.loss_manager.loss.avg

def test(loader, model, epoch):
    """

    test function for 1 epoch
        valid train
        write result data

    Args:
        loader : pytorch dataloader
        model : pytorch model
        criterion : pytorch loss function
        epoch (int) : epoch number
    """
    progress_logger.info("Test phase")
    # 各値初期化
    data_manager = my_util.DataManager("test", writer)
    model.eval()
    data_num = len(loader.dataset)  # テストデータの総数
    pbar = tqdm(total=int(data_num / opts.batch_size))
    result_text = ""
    batch_size = opts.batch_size
    hidden_dim = model.language_model.hidden_dim_encoder
    device = model.laguage_model.device
    states = (torch.zeros(1, batch_size, hidden_dim).to(device),
              torch.zeros(1, batch_size, hidden_dim).to(device))
    with torch.no_grad():
        for i, (inputs, labels, input_lengths, target_lengths, dlib) in enumerate(loader):
            # データをdeviceに載せる
            # 初期値
            input_lengths = input_lengths.to(DEVICE, non_blocking=True)
            target_lengths = target_lengths.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            inputs = inputs.to(DEVICE, non_blocking=True).float()
            dlib = dlib.to(DEVICE, non_blocking=True).float()
            outputs = model(inputs, dlib, input_lengths, False)
            # 結果保存用
            batch_size = inputs.size(0)
            # input_lengths = torch.full((1, batch_size), fill_value=outputs.size(0))
            # 統合モデルへの入力
            outputs = model(inputs, dlib, input_lengths, False, states)
            batch_size = inputs.size(0)
            # 出力とラベルのpadding
            if targets.shape[1] < outputs.shape[1]:
                targets = pad_label(targets, outputs).to(DEVICE)
            elif targets.shape[1] > outputs.shape[1]:
                outputs = pad_out(targets, outputs).to(DEVICE)
            # nn.CrossEntropyLossに入れれるようにreshape
            outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2])
            targets = targets.reshape(targets.shape[0]*targets.shape[1])
            loss = criterion(output, targets)
            for i in range(batch_size):
                output = outputs[i]
                label = labels[i]
                _, output = output.max(dim=1)
                pred = format_outputs(output)
                output = [phones[l] for l in output]
                pred = [phones[l] for l in pred]
                label = [phones[l] for l in label if phones[l] not in "_"]
                ter = my_util.calculate_error(pred, label)
                data_manager.update_acc(ter, batch_size)
                result_text += " ".join(pred) + "\n"
            result_logger.info(result_text)
            outputs = outputs.permute(1, 0, 2).log_softmax(2)
            ctc_loss = criterion(outputs, labels, input_lengths, target_lengths)
            # measure performance and record loss
            data_manager.update_loss(ctc_loss.data.item(), batch_size)
            pbar.update(1)
    pbar.close()
    data_manager.write(epoch)
    with open(f"{opts.base_path}/p-file{epoch}.txt", mode='w') as f:
        f.write(result_text)
    progress_logger.info('Test loss:{loss.avg:.4f} '
                         'Acc:{Acc.avg:4f}'.format(loss=data_manager.loss_manager.loss,
                                                   Acc=data_manager.acc_manager.total_error))

progress_logger.info("Evaluate untrained valid")
# 未学習時のモデルの性能の検証
# valid_result = valid(validloader, model, criterion, 0)

valtrack = 0
# Start Train
progress_logger.info(f"LOG : Train Started ...epoch {opts.end_epoch}まで")
scaler = amp.GradScaler(enabled=opts.amp)
for epoch in range(opts.start_epoch, opts.end_epoch + 1):
    train(trainloader, model, criterion, optimizer, scaler, epoch)
    valid_loss = valid(validloader, model, criterion, epoch)
    # save model
    is_best = valid_loss <= best_val  # ロスが小さくなったか
    # save model
    if is_best:
        valtrack = 0
        best_val = valid_loss
        best_epoch = epoch
        my_util.save_checkpoint(
            {  # modelの保存
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  #必須
                'best_val': best_val,
                'optimizer_state_dict': optimizer.state_dict(),  #必須
                "scaler": scaler.state_dict(),
                'valtrack': valtrack,
                'valid_loss': valid_loss
            },
            opts.checkpoint)
    else:
        valtrack += 1
    test(testloader, model, epoch)
    if opts.patience <= valtrack:
        break
    progress_logger.info(
        f'Validation: {valid_loss} (best:{"{0:,.5f}".format(best_val)}) (valtrack:{"{0:,.5f}".format(valtrack)})')

my_util.load_checkpoint(best_epoch, best_val, model, opts)
test(testloader, model, 1000)

writer.close()  # close tensorboard writer
progress_logger.info("Finish!!")
'''
グラフは以下で確認可能
tensorboard --logdir="./save_tb"
'''