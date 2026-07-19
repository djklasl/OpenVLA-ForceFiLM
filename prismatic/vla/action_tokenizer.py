"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins) #np.linspace生成{-1，-0.9921875，...，0.9921875，1}共256个值的一维数组
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0 #生成{（-1 + -0.9921875）/2, ..., (0.9921875 + 1)/2}共255个值的一维数组

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    # 定义 __call__ 这个“魔术方法”后，ActionTokenizer 实例就可以像函数一样被调用：
    #   at = ActionTokenizer(...);  at(action) 等价于 at.__call__(action)
    # 这里让动作 tokenizer 的用法风格和普通 tokenizer(tokenizer(text)) 一致。
    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """action——>discretized_action——>action_token_ids——>文本"""
        #np.clip函数作用：把 action 数组中所有小于 min_action 的值拉到 min_action，把所有大于 max_action 的值拉到 max_action，
        #                中间范围内的值保持原样。
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        
        #np.digitize函数：把裁剪后的连续动作数值action，查在self.bins的第几个区间里”，得到的是“每个动作落在第k个 bin 里”的整数索引k，即 0 ~ 255
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        #是一维数组，比如形状 (action_dim,)，代表“单个样本的一条动作序列
        #先把离散后的动作 bin 索引（discretized_action）映射到词表最后n_bins个 token id，再用tokenizer.decode把这一串 token id 解码成一个字符串，作为“动作 token 序列”的表示。
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            #discretized_action 至少是二维数组，比如 (batch_size, action_dim)，代表“一批样本，每个都有一条动作序列”。
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """action_token_ids->discretized_actions-->bin_centers[discretized_actions]
        将离散的动作 token ID 解码为离散动作。

        NOTE =>> 由于动作是相对于 bins（分箱边界）而不是 bin centers（箱中心）来离散化的，
                 digitization（np.digitize）返回的 bin 索引范围是 [1, #bins]（包含两端），
                 但实际上只有 (#bins - 1) 个 bin 区间。

                 因此，如果 digitization 返回了最后一个可能的索引，我们会把它映射到最后一个 bin 区间。

        EXAMPLE =>> 假设 self._bins 有 256 个值，那么 self._bin_centers 只有 255 个值。
                    digitization 返回的索引范围是 [1, 256]。
                    我们先把所有索引减 1，使它们变为 [0, 255]。此时仍然有一个索引（i == 255），
                    如果直接用来索引 self._bin_centers 就会越界，因为 self._bin_centers 只有 255 个元素
                    （合法索引是 0~254）。因此，当 i == 255 时，我们再把它减 1，让它变成最后一个 bin center
                    的索引。实际实现时更简单：直接把索引 clip 到 [0, 255 - 1] 这个区间即可。
        """
        #把 token id 还原成 bin 的索引 大致在 [1, n_bins]）
        discretized_actions = self.tokenizer.vocab_size - action_token_ids

        #把索引范围从 [1, n_bins] 映射到 [0, n_bins - 2]，以防止越界
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
