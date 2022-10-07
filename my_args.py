# -*- coding: utf-8 -*-

import os
import shutil
import multiprocessing
import argparse

def get_parser(time: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Set parameter')
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--dir', default=current_file_path, type=str, help='directory')
    parser.add_argument('--debug', default=True, type=bool, help="debug mode")
    parser.add_argument('--message', default='', type=str, help="message")
    parser.add_argument('--mode', default='FV', type=str, help="mode")
    parser.add_argument('--logger-path', default=f"build/{time}/", type=str, help="logger build path")
    # general
    parser.add_argument('--workers', default=int(0), type=int, help="使用するCPUコア数")
    # cudnnの自動チューナー:場合によって速くなったり遅くなったり(入力サイズが常に一定だといいらしいが)
    parser.add_argument('--benchmark', default=False, type=bool, help="torch benchmark")
    parser.add_argument('--amp', default=False, type=bool, help="amp")
    # Dataset path
    parser.add_argument('--label',
                        default="data/train/label/*",
                        type=str,
                        help="label")
    parser.add_argument('--image-path',
                        default='data/train/tensor/*',
                        type=str,
                        help="train image path")
    parser.add_argument('--csv-path',
                        default='data/train/csv/*',
                        type=str,
                        help="train csv path")

    # result path
    parser.add_argument('--no_check', action='store_true', help="ファイル削除を尋ねるか")
    parser.add_argument('--checkpoint', default=f'build/{time}/checkpoint/', type=str, help="checkpoint path")
    parser.add_argument('--tensorboard', default=f'build/{time}/tensorboard/', type=str, help="tensorboard path")
    parser.add_argument('--logger-config-path', default=f'assets/log_config.json', type=str, help="logger config path")
    parser.add_argument('--base-path', default=f"build/{time}/", type=str, help="logger build path")
    # model
    parser.add_argument('--lstm-layer', default=2, type=int, help="LSTM layer")
    parser.add_argument('--lstm-hidden', default=100, type=int, help="LSTM hidden size")
    parser.add_argument('--batch_size', default=8, type=int, help="batch size")
    # training
    parser.add_argument('--train-size', default=(0, 7486), type=tuple, help="train-size")
    parser.add_argument('--valid-size', default=(7486, 7586), type=tuple, help="valid-size")
    parser.add_argument('--test-size', default=(7586, 7596), type=tuple, help="test-size")
    parser.add_argument('--patience', default=25, type=int, help="patience")
    parser.add_argument('--end_epoch', default=40000, type=int, help="epoch")  # d: 720
    parser.add_argument('--start_epoch', default=0, type=int, help="start epoch")
    parser.add_argument('--resume', default='', type=str, help="resume checkpoint path")
    parser.add_argument('--token', default=f"build/{time}/token.json", type=str, help="token path")
    # Optimizer
    parser.add_argument('--lr', default=1e-4, type=float, help="学習率")  # d: 0.0001
    parser.add_argument('--momentum', default=0.9, type=float, help="モメンタム")
    parser.add_argument('--weight_decay', default=0, type=float, help="weight decay")
    parser.add_argument('--clip', default=5.0, type=float, help="")
    # other
    parser.add_argument('--seed', default=2, type=int, help="randomのseed")
    parser.add_argument('--cpu', action='store_true', help="cpuで動作させたい場合")
    parser.add_argument('--device', default="cuda:0", help="device")

    return parser

def set_debug_mode(opts: argparse.Namespace, log_conf: dict) -> None:
    """

    Turn on debug mode
    Don't make new tensorboard / checkpoint

    """
    shutil.rmtree("build/debug", ignore_errors=True)
    opts.base_path = "build/debug"
    opts.tensorboard = "build/debug/tensorboard"
    opts.checkpoint = "build/debug/checkpoint"
    opts.token = f"build/debug/token.json"
    os.makedirs(opts.tensorboard, exist_ok=True)
    os.makedirs(opts.checkpoint, exist_ok=True)
    opts.train_size = (0, 200)
    opts.valid_size = (200, 400)
    opts.test_size = (400, 500)
    opts.batch_size = 2
    opts.no_check = True
    opts.end_epoch = 2000
    log_conf["handlers"]["fileHandler"]["filename"] = f'build/debug/progress.log'
    log_conf["handlers"]["result_fileHandler"]["filename"] = f'build/debug/result.log'

def set_release_mode(opts: argparse.Namespace, log_conf: dict) -> None:
    """

    Turn on release mode
    Make new tensorboard / checkpoint

    """
    os.makedirs(opts.base_path, exist_ok=True)
    log_conf["handlers"]["fileHandler"]["filename"] = f'{opts.base_path}/progress.log'
    log_conf["handlers"]["result_fileHandler"]["filename"] = f'{opts.base_path}/result.log'
