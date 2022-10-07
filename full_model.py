import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from phoneme_dataset import PAD_NUM

class SSRWFullModel(nn.Module):
    """
    SSRW用の最終的なモデル構造です。
    服部先輩の読唇モデルと、青山or加藤先輩の言語モデルをくっつけただけです。
    """
    def __init__(self,
                 lip_reading_model: nn.Module,
                 language_model: nn.Module,
                 phoneme_dict: dict,
                 device=torch.device('cuda:0')):
        """
        コンストラクタです。3つ渡してください。
        1. 読唇モデル(nn.Module型)
        2. 言語モデル(nn.Module型)
        3. 音素の辞書(dict型)

        Args:
            lip_reading_model (nn.Module): 読唇モデル
            language_model (nn.Module): 言語モデル
            phoneme_dict (dict): 音素辞書
            device (str or torch.device, optional): 一応deviceもください. Defaults to torch.device('cuda:0').
        """
        super(SSRWFullModel, self).__init__()
        self.lip_reading_model = lip_reading_model
        self.language_model = language_model
        self.dict = phoneme_dict
        self.device = device
    
    def forward(self,
                x: torch.Tensor,
                dlib: torch.Tensor,
                input_legth: torch.Tensor,
                data_aug: bool,
                h: tuple) -> tuple:
        """
        nn.Moduleからの必須オーバライド
        xは画像、dlibはdlib特徴量, hは言語モデルでlstmを使う場合のhです。
        デフォルトでオーバロードをサポートしてないpythonはクソ

        Args:
            x (torch.Tensor): 入力画像系列のバッチ
            dlib (torch.Tensor): 入力dlib特徴量系列のバッチ
            h (tuple): LSTM用のhidden feature

        Returns:
            tuple: 返すのはx: torch.Tensorとh: (torch.Tensor, torch.Tensor)のtuple
        """
        x = self.lip_reading_model(x, dlib, input_legth, data_aug)
        x_sentences = []
        x = torch.argmax(x, dim=2)
        for _x in x:
            x_sentences.append(torch.Tensor(format_outputs(_x)))
        x = pad_sequence(x_sentences, padding_value=PAD_NUM, batch_first=True).to(dtype=torch.long, device=self.device)
        print(x.device, h[0].device)
        x, h = self.language_model(x, h)
        return x, h

# format outputs
def format_outputs(verbose):
    predict = [verbose[0]]
    for i in range(1, len(verbose)):
        if verbose[i] != predict[-1]:
            predict.append(verbose[i])
    predict = [l for l in predict if l != PAD_NUM]
    return predict