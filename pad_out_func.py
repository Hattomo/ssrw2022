import torch
from phoneme_dataset import PAD_NUM, MASK_NUM

def pad_out(labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    labelの方が長い場合にoutputsをpaddingする関数
    outputsをpaddingする際は以下のようにPAD_NUMの部分だけ1にして後は0で
    [[[0., 0., 0., 1., 0., ...],
      [0., 0., 0., 1., 0., ...],
      ... Seq_length
      [0., 0., 0., 1., 0., ...]
    ],
    ... BATCH_SIZE
     [[0., 0., 0., 1., 0., ...],
      [0., 0., 0., 1., 0., ...],
      ... Seq_length
      [0., 0., 0., 1., 0., ...]
     ]
    ]

    Args:
        labels (torch.Tensor): [BATCH_SIZE, Seq_length_A]のTensor
        outputs (torch.Tensor): [BATCH_SIZE, Seq_length_B, Vocab_size]のTensor

    Returns:
        torch.Tensor: [BATCH_SIZE, Seq_length_A, Vocab_size]
                      paddingされた後のoutputsのTensor
    """
    pass
