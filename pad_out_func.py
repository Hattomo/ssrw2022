import torch
from phoneme_dataset import PAD_NUM, MASK_NUM
from torch.nn.functional import one_hot

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

    labels_length = labels.shape[1]
    outputs_length = outputs.shape[1]

    batch_size = labels.size(0)
    #print(batch_size)
    if labels_length > outputs_length:
      padding_length = labels_length - outputs_length
      #print(padding_length)
      pad = torch.full(size=(batch_size, padding_length), fill_value=PAD_NUM)
      pad = one_hot(pad)
      outputs = torch.cat([outputs,pad],dim = 1)
      outputs_length = outputs.shape[1]
      #print(labels.shape,outputs.shape)
      #print(labels)
      #print(outputs)
      assert labels_length == outputs_length, "labelとoutputの長さが違うよ"
    return outputs

if __name__ == "__main__":
    PAD_NUM = 2
    labels = torch.tensor([[1, 2, 0, 1, 2],[1, 2, 0, 1, 2]])
    outputs = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1],[1, 0, 0]]])
    pad_out(labels, outputs)
