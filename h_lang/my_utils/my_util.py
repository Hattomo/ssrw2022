# -*- coding: utf-8 -*-
"""
my_util.py
関数のまとめ
"""
import shutil, os
import numpy as np
import copy

import torch
from torch.utils.tensorboard import writer
import torchmetrics

def check_file(path: str) -> None:
    """
    check file exists
    """
    if os.path.exists(path):
        check = input("CHECK : {} is already exists. Delete them?(y/n) : ".format(path))
        if check == "y" or check == "Y":
            shutil.rmtree(path)
            print("FILE : ", path, "was deleted")
        else:
            print("INFO : Aborted")
            exit()

# save model parameters
def save_checkpoint(state, dir_path)->None:
    filename = os.path.join(dir_path, 'epoch%03d_val%.3f.pth' % (state['epoch'], state['best_val']))
    torch.save(state, filename)

def load_checkpoint(best_epoch,best_val,model,opts)->None:
    filename = os.path.join(opts.checkpoint, 'epoch%03d_val%.3f.pth' % (best_epoch, best_val))
    model.load_state_dict(
        torch.load(filename)['model_state_dict'])

# 平均と現在の値を計算して保存するクラス
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

class LossManager(object):

    def __init__(self, mode: str, writer: writer):
        self.loss = AverageMeter()
        self.mode = mode
        self.writer = writer

    def update(self, loss, batch_size):
        self.loss.update(loss, batch_size)

    def write(self, epoch):
        self.writer.add_scalars(f"loss", {self.mode: self.loss.avg}, epoch)

class TokenErrorRate():
    """
    Member of TER
    """

    def __init__(self, total_error: float, subtitle_error: float, delete_error: float, insert_error: float,
                 len_ref: float) -> None:
        self.total_error = total_error
        self.substitute_error = subtitle_error
        self.delete_error = delete_error
        self.insert_error = insert_error
        self.len_ref = len_ref

class TokenErrorRateManager():
    """
    Calculate TER

    Func:
        reset() : initialize ter
        update(ter: TokenErrorRate) : update ter
        write(epoch: int) : write ter to tensorboard
    """

    def __init__(self, mode: str, writer) -> None:
        self.reset()
        self.writer = writer
        self.mode = mode

    def reset(self):
        self.total_error = AverageMeter()
        self.substitute_error = AverageMeter()
        self.delete_error = AverageMeter()
        self.insert_error = AverageMeter()
        self.len_ref = AverageMeter()

    def update(self, ter: TokenErrorRate, batch_size: int) -> None:
        self.total_error.update(ter.total_error, batch_size)
        self.substitute_error.update(ter.substitute_error, batch_size)
        self.delete_error.update(ter.delete_error, batch_size)
        self.insert_error.update(ter.insert_error, batch_size)
        self.len_ref.update(ter.len_ref, batch_size)

    def write(self, epoch: int) -> None:
        self.writer.add_scalars(f'ter/total_error', {self.mode: self.total_error.avg}, epoch)
        self.writer.add_scalars(f'ter/substitute_error', {self.mode: self.substitute_error.avg}, epoch)
        self.writer.add_scalars(f'ter/delete_error', {self.mode: self.delete_error.avg}, epoch)
        self.writer.add_scalars(f'ter/insert_error', {self.mode: self.insert_error.avg}, epoch)
        self.writer.add_scalars(f'ter/len_ref', {self.mode: self.len_ref.avg}, epoch)

class DataManager():

    def __init__(self, mode: str, writer: writer) -> None:
        self.loss_manager = LossManager(mode, writer)
        self.acc_manager = TokenErrorRateManager(mode, writer)
        self.writer = writer

    def update_loss(self, loss: float, batch_size: int) -> None:
        self.loss_manager.update(loss, batch_size)

    def update_acc(self, acc: TokenErrorRate, batch_size: int) -> None:
        self.acc_manager.update(acc, batch_size)

    def reset(self) -> None:
        self.loss_manager.reset()
        self.acc_manager.reset()

    def write(self, epoch: int) -> None:
        self.acc_manager.write(epoch)
        self.loss_manager.write(epoch)

def calculate_error(hypothesis, reference):
    ''' レーベンシュタイン距離を計算し，
        置換誤り，削除誤り，挿入誤りを出力する
    hypothesis:       認識結果(トークン毎に区切ったリスト形式)
    reference:        正解(同上)
    total_error:      総誤り数
    substitute_error: 置換誤り数
    delete_error:     削除誤り数
    insert_error:     挿入誤り数
    len_ref:          正解文のトークン数
    '''
    # 認識結果および正解系列の長さを取得
    len_hyp = len(hypothesis)
    len_ref = len(reference)

    # 累積コスト行列を作成する
    # 行列の各要素には，トータルコスト，
    # 置換コスト，削除コスト，挿入コストの
    # 累積値が辞書形式で定義される．
    cost_matrix = [[{"total": 0,
                     "substitute": 0,
                     "delete": 0,
                     "insert": 0} \
                     for j in range(len_ref+1)] \
                         for i in range(len_hyp+1)]

    # 0列目と0行目の入力
    for i in range(1, len_hyp + 1):
        # 縦方向への遷移は，削除処理を意味する
        cost_matrix[i][0]["delete"] = i
        cost_matrix[i][0]["total"] = i
    for j in range(1, len_ref + 1):
        # 横方向への遷移は，挿入処理を意味する
        cost_matrix[0][j]["insert"] = j
        cost_matrix[0][j]["total"] = j

    # 1列目と1行目以降の累積コストを計算していく
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            #
            # 各処理のコストを計算する
            #
            # 斜め方向の遷移時，文字が一致しない場合は，
            # 置換処理により累積コストが1増加
            substitute_cost = \
                cost_matrix[i-1][j-1]["total"] \
                + (0 if hypothesis[i-1] == reference[j-1] else 1)
            # 縦方向の遷移時は，削除処理により累積コストが1増加
            delete_cost = cost_matrix[i - 1][j]["total"] + 1
            # 横方向の遷移時は，挿入処理により累積コストが1増加
            insert_cost = cost_matrix[i][j - 1]["total"] + 1

            # 置換処理，削除処理，挿入処理のうち，
            # どの処理を行えば累積コストが最も小さくなるかを計算
            cost = [substitute_cost, delete_cost, insert_cost]
            min_index = np.argmin(cost)

            if min_index == 0:
                # 置換処理が累積コスト最小となる場合

                # 遷移元の累積コスト情報をコピー
                cost_matrix[i][j] = \
                    copy.copy(cost_matrix[i-1][j-1])
                # 文字が一致しない場合は，
                # 累積置換コストを1増加させる
                cost_matrix[i][j]["substitute"] \
                    += (0 if hypothesis[i-1] \
                        == reference[j-1] else 1)
            elif min_index == 1:
                # 削除処理が累積コスト最小となる場合

                # 遷移元の累積コスト情報をコピー
                cost_matrix[i][j] = copy.copy(cost_matrix[i - 1][j])
                # 累積削除コストを1増加させる
                cost_matrix[i][j]["delete"] += 1
            else:
                # 置換処理が累積コスト最小となる場合

                # 遷移元の累積コスト情報をコピー
                cost_matrix[i][j] = copy.copy(cost_matrix[i][j - 1])
                # 累積挿入コストを1増加させる
                cost_matrix[i][j]["insert"] += 1

            # 累積トータルコスト(置換+削除+挿入コスト)を更新
            cost_matrix[i][j]["total"] = cost[min_index]

    #
    # エラーの数を出力する
    # このとき，削除コストは挿入誤り，
    # 挿入コストは削除誤りになる点に注意．
    # (削除コストが1である
    #    = 1文字削除しないと正解文にならない
    #    = 認識結果は1文字分余計に挿入されている
    #    = 挿入誤りが1である)
    #

    # 累積コスト行列の右下の要素が最終的なコストとなる．
    total_error = cost_matrix[len_hyp][len_ref]["total"]
    substitute_error = cost_matrix[len_hyp][len_ref]["substitute"]
    # 削除誤り = 挿入コスト
    delete_error = cost_matrix[len_hyp][len_ref]["insert"]
    # 挿入誤り = 削除コスト
    insert_error = cost_matrix[len_hyp][len_ref]["delete"]

    # 各誤り数と，正解文の文字数
    # (誤り率を算出する際に分母として用いる)を出力
    ter = TokenErrorRate(total_error, substitute_error, delete_error, insert_error, len_ref)
    return ter

def calculate_error2(preds, target):
    return torchmetrics.WordErrorRate(preds, target)
