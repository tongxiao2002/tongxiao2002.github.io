---
title: T5 Decode training 流程详解
date: 2023-05-05 20:20:00 +0800
categories: [Deep Learning]
tags: [transformer, decode]
---

# T5 Decode training 流程详解

## Intro-Introduction

（好像鸽了好久没写 blog 了......）

## Introduction

最近因为科研需要，得对 T5 模型的 decode 过程十分了解（其实本质上几乎就是所有基于 Transformer 的模型的 decode 过程），基于对自己金鱼记忆的充分认识，决定写下来一些关键的步骤。

基于 Transformer 模型的 decode 大致流程有着极多文章都已经讨论过了，无非就是自回归的、自左向右的方式进行 decode，其中有一篇讲的极好的 [blog](https://jalammar.github.io/illustrated-transformer/)。但是这些文章讲的都是非常原理性的东西，而没有深入地结合代码实现进行讲解，因此在实际动手时未免会有不少疑惑。比如我看这些 blog 的感觉就是重复不断地调用 decoder 的 forward 函数，直到生成终止符，但是 huggingface-T5 的实现代码却是仅调用一次 forward 函数就生成了所有的 token，这让我很疑惑，因此决定把 decode 过程中所有细节都搞明白并写下来。

（写完之后的后记：写完之后梳理了一下才反应过来 transformer decode 在 training 和 inference 阶段的实现逻辑大相径庭，本文应当限制在 training 阶段的 transformer decode）

## 代码实现

这篇文章以 [huggingface-T5ForConditionalGeneration](https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/t5#transformers.T5ForConditionalGeneration) 为例子，我猜测类似的 BART，GPT 等模型的 decode 过程应该也差不多。

我们假设用以下方式调用 `T5ForCondiitonalGeneration`：其实就是 huggingface 给的例子。

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# training
input_ids = tokenizer(
    "The <extra_id_0> walks in <extra_id_1> park",
    return_tensors="pt"
).input_ids
labels = tokenizer(
    "<extra_id_0> cute dog <extra_id_1> the <extra_id_2>",
    return_tensors="pt"
).input_ids
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

# inference
input_ids = tokenizer(
    "summarize: studies have shown that owning a dog is good for you",
    return_tensors="pt"
).input_ids  # Batch size 1
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

废话不多说直接看 [forward](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/t5/modeling_t5.py#L1617) 函数。可以看到一堆输入参数看的头疼，不过无所谓，我们的例子中只有 `input_ids` 和 `labels`，我们先直接看函数体。

### encoder 部分

跳过函数体刚开始的几行设置判断，可以看到 forward 函数首先要做的就是对输入进行 encode，这里涉及到两个输入：`input_ids` 和 `encoder_outputs`。若 `encoder_outputs` 在输入时已经提供，则不会使用 T5 的 encoder 进行编码，decoder 的部分输入将直接源于 `encoder_outputs`；否则将使用 T5 的 encoder 对 `input_ids` 进行编码，从而得到 `encoder_outputs`。最终取 `encoder_outputs` 中最后一层的 hidden_state 用作 decoder 的输入。

至于 encoder 的编码具体细节，我们这篇文章就不细谈了，感觉只是普通的双向编码过多层 transformer，比较简单易懂。

### decoder 部分

decoder 部分从[构建输入](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/t5/modeling_t5.py#L1700)开始，这里的逻辑是，若用户没有显式提供 `decode_input_ids` 和 `decode_inputs_embeds` 但是提供了 `labels` 的话，则将使用 `labels` 构建出 `decode_input_ids`，这是因为后面 `self.decoder` 的输入需要 `decoder_input_ids` 或 `decoder_inputs_embeds` 至少一个。（话说为什么需要 `decode_input_ids` ？）具体通过 `_shift_right` 函数实现。

#### `_shift_right`

[`_shift_right`](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/t5/modeling_t5.py#L847) 函数用于从 labels 构建出 decoder 的输入。具体逻辑为在 labels 的最前面加上 `decoder_start_token_id` 这么一个 token，用于指示开始生成，同时去掉最后一个 token，从而构成了 decoder 的输入 `decoder_input_ids`。**（这里最后一个 token 是什么？）**

后面紧跟着的就是一些 `model_parallel` 的设置，不难理解这是一些提高模型训练性能的代码，与具体功能无关，直接跳过，从而来到了 `self.decoder` 部分。这里 `self.decoder` 其实就是 `T5Stack` 模块，因此现在执行的其实是 `T5Stack` 模块的 forward 函数。

#### `T5Stack.forward`

现在来看 [T5Stack.forward](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/t5/modeling_t5.py#L943)，同样的，刚开始都是一些设置性的代码，然后紧跟着的就是对 `input_ids` 和 `inputs_embeds` 的判断，二者必须有且只有一个不为空，这一点其实我们在 `T5ForConditionalGeneration.forward` 中就已经保证了。

##### `get_extended_attention_mask`

下一步就是根据输入的 `input_ids` 和 `encoder_hidden_states` 的长度设置相应的 `attention_mask`。`attention_mask` 里面 `1` 为有效 token，`0` 为被 mask 的 token。

此时注意到下一行为：

```python
extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

def get_extended_attetion_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device = None):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
```

根据 huggingface 对函数 `get_extended_attention_mask` 的注释，可以看出就是该函数对 decoder 实现了自左到右的 decode 方式，具体实现方式为对当前待 decode 的 token 之后的所有 token 进行 mask。

由于我们输入的 `attention_mask` 为 2 维，因此直接到了：

```python
extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
    input_shape, attention_mask, device
)
```

根据 huggingface 的注释，该函数是真正用于应用 `casual mask` 的函数，该函数用了一个很 trick 的方式生成了 bool 型下三角阵：

```python
batch_size, seq_length = input_shape
seq_ids = torch.arange(seq_length, device=device)
# 非常的 trick 啊
causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]

# 生成结果样例
[[ True, False, False, False, False, False, False, False, False, False],
 [ True,  True, False, False, False, False, False, False, False, False],
 [ True,  True,  True, False, False, False, False, False, False, False],
 [ True,  True,  True,  True, False, False, False, False, False, False],
 [ True,  True,  True,  True,  True, False, False, False, False, False],
 [ True,  True,  True,  True,  True,  True, False, False, False, False],
 [ True,  True,  True,  True,  True,  True,  True, False, False, False],
 [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
 [ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
 [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]]
```

下面讨论了当 `causal_mask` 的长度比 `attention_mask` 长度短的情况。那么为什么会有这种情况？这两者的长度是什么？其实这一点在 [T5Stack.forward](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/t5/modeling_t5.py#L990) 中有体现，可以看到这一行中 `mask_seq_length` 的长度为 `seq_length` 加上了 `past_key_values` 的长度，而 `seq_length` 就是 `causal_mask` 的长度。（这里 `past_key_values` 的长度是什么？我猜测可能是类似于 encoder 的输入，也就是说对于每一步 decode 而言都是有效信息）因此若 `mask_seq_length` 大于 `seq_length` 则在前面补 `1`（即为有效 token），补到与 `attention_mask` 最后一维相等长度。最后将 `causal_mask` 和 `attention_mask` 作逻辑且操作，即都为有效 token 最终才为有效 token，否则该 token 被 mask 掉。同时整个 `extended_attention_mask` 扩展至 4 维，第二维为 `1`。

现在回到 `get_extended_attetion_mask` 函数，最后有几行：

```python
# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
# masked positions, this operation will create a tensor which is 0.0 for
# positions we want to attend and -10000.0 for masked positions.
# Since we are adding it to the raw scores before the softmax, this is
# effectively the same as removing these entirely.
extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
return extended_attention_mask
```

其实就是将原来的：`1` 表示有效token，`0` 表示被 mask，改成了现在的：`0` 表示有效token，`-10000.0` 表示被 mask。原因在注释中也解释的很详细：后续直接将 mask 值加到 score 上再进行 Softmax 操作，因此 score 加上 `-10000` 的 mask 经 Softmax 操作后自然是接近于 0，从而被 mask。最终返回 `extended_attention_mask`，我们又回到了 `T5Stack.forward`。

回到 `T5Stack.forward` 之后第一件事就是生成 `encoder_hidden_states` 的 `encoder_extended_attention_mask`，这个 mask 和之前的一样，也是需要改成：`0` 表示有效 token，`-10000.0` 甚至更多表示被 mask，原因也是和之前一样的。由于来自 encoder 的信息可以全部利用，因此不需要像 `causual_mask` 中一样生成一个下三角阵，除 `pad_token` 外全部为有效即可。

##### `get_head_mask`

继续向下走，到了这一行：

```python
head_mask = self.get_head_mask(head_mask, self.config.num_layers)
cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)

def get_head_mask(self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False):
    """
    Prepare the head mask if needed.

    Args:
        head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
        num_hidden_layers (`int`):
            The number of hidden layers in the model.
        is_attention_chunked: (`bool`, *optional*, defaults to `False`):
            Whether or not the attentions scores are computed by chunks or not.

    Returns:
        `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
        `[None]` for each layer.
    """
    if head_mask is not None:
    head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
    if is_attention_chunked is True:
        head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers

    return head_mask
```

这一行用于生成 `head_mask`，我们的例子中由于 `head_mask` 为 `None`，因此返回值就是一个长度为 `num_hidden_layers` 的、全都为 `None` 的 list。

接着准备好一系列用于接收中间信息的元组，我们就终于开始遍历 forward 各个 transformer block 了。（终于）

简单起见，我们依然跳过所有的 `model_parallel` 部分以及 `gradient_checkpoint` 部分，直接来到[遍历 module](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/t5/modeling_t5.py#L1086) 的部分。显然，我们又要进入 `T5Block.forward` 函数了。。。

##### `T5Block.forward`

由于我们 `past_key_value` 都为 `None`，因此直接开始对 `decoder_inputs` 进行 `self-attention`，这里的 `hidden_states` 其实就是之前的 `decoder_input_ids` 经 embedding 之后的向量。**注意到我们之前得到的 4 维 `causual_mask` 也在这个时候输入到 `self-attention` 中使用，与我之前所想不同的是，在训练过程中，由于 ground_truth 已知，因此模型生成时并不会每次生成一个 token 后再生成下一个，而是一句话内不同的位置，看到的词的数量不一样，比如第一个词就只能看第一个词，而第二个词能看前两个，然后在一次迭代中所有位置都预测相应的词，这和我们之前得到的 `causal_mask` 矩阵的形式是对应的，而在 inference 中，则使用 `modeling_utils` 中的函数 `generate`，这就是生成一个词之后再生成下一个词，和训练时的生成逻辑是不同的。**然后由于我们是 decoder，因此会将 `encoder_hidden_states` 和 `self-attention` 之后的 `hidden_states` 进行 `cross-attention`，这两步都和标准的 `transformer-decoder` 结构是完全符合的。

最终 `T5Block.forward` 的返回结果分为 3 块：经 `cross-attention` 以及 FFN 之后的 `hidden_states`，经 `self-attention` 和 `cross-attention` 后的两个相加的 `present_key_value_state` 以及经`self-attention` 和 `cross-attention` 后的两个 `attention_outputs`。

现在又回到 `T5Stack.forward`。注意到 `position_bias` 和 `encoder_decoder_position_bias` 这两个本来会被用于输入 `T5Block.forward` 的两个参数会被改变，这也与 huggingface 的注释：这两个参数在不同层之间共享，是相符的，其他的输出就是加入到最终输出中。

遍历 block 结束后，`T5Stack.forward` 也结束了，最终会返回：最后一层 `T5Block` 的 `hidden_state`，所有层的 `present_key_value_states`，所有层的中间状态 `all_hidden_states`，所有层的 `attention_outputs` 和 `cross_attention_outputs`。

现在回到了 [`T5ForConditionalGeneration.forward`](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/t5/modeling_t5.py#L1731) 中。

取最后一层的 `hidden_states` 作为 `sequence_output`，将该 `sequence_output` 过一层 `lm_head`，用于预测下一个词。

至此，`T5ForConditionalGeneration` 的 decode 部分完成。

## 总结

总而言之，整个 transformer decode 过程是很复杂的，我们还略过了很多性能部分，或者是由于我们的样例很简单，因此还有不少输入参数没有考虑到。但是总的流程差不多了解了，尤其是怎么实现从左到右的生成（或者说是 mask），以及训练过程中的生成模式并不是一个一个生成，而是一次迭代中不同位置都在预测不同的词，从而能够再一次迭代中完全整个序列的预测 & 训练。

本文完全是按照本人自己的理解写下的，我在写的过程中就感觉有些地方表达的不太合适或者不太知道该如何表达。鉴于估计没人看所以写成自己看得懂的形式也就差不多了（，如果真有人认真看了但是感觉写的依托答辩还请您直接邮件联系我来仔细讨♂论讨♂论。

立个小 flag，下次更新 `generate` 函数的工作原理，也就是 transformer decode 在 inference 是怎么具体实现的。

