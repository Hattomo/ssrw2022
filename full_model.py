import torch
import torch.nn as nn

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
    
    def forward(self, x: torch.Tensor, dlib: torch.Tensor, h: tuple) -> tuple:
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
        x = self.lip_reading_model(x, dlib)
        x = torch.argmax(x, dim=2)
        x, h = self.language_model(x, h)
        return x, h