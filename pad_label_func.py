import torch
from phoneme_dataset import PAD_NUM, MASK_NUM
from torch.nn import functional as F

def pad_label(labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    outputsの方が長い場合にlabelsをpaddingする関数
    PAD_NUMで埋めてください。

    Args:
        labels (torch.Tensor): [BATCH_SIZE, Seq_length_A]のTensor
        outputs (torch.Tensor): [BATCH_SIZE, Seq_length_B, Vocab_size]のTensor

    Returns:
        torch.Tensor: [BATCH_SIZE, Seq_length_B]
                      paddingされた後のlabelのTensor
    """

    batch_size = labels.size(0)
    if labels.size(-1) < outputs.size(-2):
        padding_length = outputs.size(-2) - labels.size(-1)
        pad = torch.full(size=(batch_size, padding_length), fill_value=PAD_NUM)
        labels = torch.cat([labels, pad], dim=1)
    return labels

if __name__ == "__main__":
    PAD_NUM = 2
    labels = torch.tensor([[1, 2],[1, 2]])
    outputs = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1],[1, 0, 0]]])
    pad_label(labels, outputs)
