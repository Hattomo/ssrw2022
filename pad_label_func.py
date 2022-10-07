import torch
from phoneme_dataset import PAD_NUM, MASK_NUM

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
    pass
