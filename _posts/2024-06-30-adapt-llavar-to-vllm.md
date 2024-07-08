---
title: vllm 推理适配自定义模型 (1)
date: 2024-06-30 16:30:00 +0800
categories: [Transformers, Large Language Models]
tags: [vllm, inference]
---

本文对应代码：[https://github.com/tongxiao2002/vllm-for-LMMs](https://github.com/tongxiao2002/vllm-for-LMMs)

## 背景

最近因为科研需求，需要测试一些多模态大语言模型（LMM）在某些任务上的性能。由于是“大”语言模型，免不了需要多卡推理（其实后来发现也不需要，13B 模型完全可以正常在一张 A800-80G 卡上跑...），因此就尝试了好几个分布式训练 or 推理框架，包括 `accelerate`，`deepspeed-inference` 以及今天的主角 `vllm`。我最开始想到的就是 `vllm`，因为之前用 `vllm` 做 LLM 的推理体验非常好，想着 LMM 相比 LLM 就加个图而且在计算过程中也是当作 token 处理，应该差别不大。然而看了眼 `vllm` 的 vlm 相关文档感觉很不详细，而且有很多我需要测试的 LMM 并不支持，包括 `LLaVAR`， `Qwen-VL` 等等。所以又回过头去试了试 `accelerate` 以及 `deepspeed-inference`，虽然这两兄弟很好实现分布式推理，但是速度实在太慢，同样的 `llava-1.5-7b-hf` 和 600 条数据，`deepspeed-inference` 需要跑 7h，`accelerate` 就更不用说了，而 `vllm` 只需要 4min，速度差实在是太夸张了（也可能是我 `deepspeed` 和 `accelerate` 用的不对🤔）。所以最后还是狠下心回来啃 `vllm` 源代码尝试自己做适配。

## 官方文档

`vllm` 官方也给出了一个非常粗略的适配自定义模型的[文档](https://docs.vllm.ai/en/latest/models/adding_model.html)，但看这意思其实还是得自己啃源代码然后自己改才能适配，并没有提供一个用户友好的接口。

官方文档给出的添加自定义模型的步骤可以分为 4 步：

1. 将自定义模型的 `forward` 函数接口（通常为 `huggingface-transformers` 的接口）改为 `vllm` 的通用接口：

    ```python
    def forward(
        self,
        input_ids: torch.Tensor,
    -    attention_mask: Optional[torch.Tensor] = None,
    -    position_ids: Optional[torch.LongTensor] = None,
    -    past_key_values: Optional[List[torch.FloatTensor]] = None,
    -    inputs_embeds: Optional[torch.FloatTensor] = None,
    -    labels: Optional[torch.LongTensor] = None,
    -    use_cache: Optional[bool] = None,
    -    output_attentions: Optional[bool] = None,
    -    output_hidden_states: Optional[bool] = None,
    -    return_dict: Optional[bool] = None,
    -) -> Union[Tuple, CausalLMOutputWithPast]:
    +    positions: torch.Tensor,
    +    kv_caches: List[torch.Tensor],
    +    attn_metadata: AttentionMetadata,
    +) -> Optional[SamplerOutput]:
    ```

    那现在接口参数就仅有 `input_ids`, `positions`, `kv_caches` 和 `attn_metadata`。
2. （可选）将自定义模型中的 Linear Layer 以及 Embedding 等改为支持 tensor paralellism 的形式，比如 `vllm` 中提供的 `QKVParallelLinear`, `VocabParallelEmbedding` 等等，不然没法通过 tensor parallelism 进行多卡推理加速。如果模型本身特别大，比如 `Llama-3-70B` 这种，那就必须要实现这一步，不然一张卡塞不下整个模型。
3. 重写 `load_weights` 函数，用于从 checkpoints 中加载参数到模型中。
4. 注册模型，让 `vllm.LLM` 能够识别自定义模型并执行：
    ```python
    from vllm import ModelRegistry
    from your_code import YourModelForCausalLM
    ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
    ```
这四步看起来最难的其实就是1、2步，因为涉及到模型结构和 `forward` 逻辑的重写。我今天以适配 `LLaVAR` 为例进行说明，由于 `llava-1.5` 模型是 `vllm` 原生支持的模型，`LLaVAR` 与 `LLaVA` 的区别主要在于参数的加载形式不同以及一点模型结构上的区别，绝大部分模型结构和 `forward` 逻辑是相同的，因此本文将主要涉及第3、4步，第1、2步将留到后面我需要实现一个 `vllm` 完全不支持的模型再写。

本文行文顺序大致上会跟官方文档上写的 4 个步骤保持一致，但是由于很多其他问题的存在，因此会加入一些其他必要步骤。

LLaVAR Github Repo: [https://github.com/SALT-NLP/LLaVAR](https://github.com/SALT-NLP/LLaVAR)

LLaVAR HuggingFace Repo: [https://huggingface.co/SALT-NLP/LLaVAR_delta](https://huggingface.co/SALT-NLP/LLaVAR_delta)

## 实现 `LlavaRForConditionalGeneration` 类，修改模型结构

如上文所述，`LLaVAR` 绝大部分与 `llava-1.5` 相同，因此直接继承 `LlavaForConditionalGeneration` 类，但是要注意的是，这里继承的应当是 `vllm` 中实现的类，否则还是得改 `forward` 逻辑和模型结构，属于多此一举。

`vllm` 中 `LlavaForConditionalGeneration` 类具体实现：[link](https://github.com/vllm-project/vllm/blob/0f0d8bc065f3608e7657a9696f5d2d7c0d6722d1/vllm/model_executor/models/llava.py#L90)

```python
# llavar.py

from vllm.model_executor.models.llava import LlavaForConditionalGeneration
class LLavaRForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(
        self,
        config: LlavaConfig,
        vision_language_config: VisionLanguageConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None
    ) -> None:
        super().__init__(config, vision_language_config, cache_config, quant_config)
```

由于 `LlavaForConditionalGeneration` 类实现的是 `llava-1.5`，使用了双层 MLP 作为 multimodal projector，而 `LLaVAR` 作为基于 `llava-1.3` 的模型，仅用了一层 Linear Layer 作为 multimodal projector，因此需要对这部分模型结构进行改变，即删除 `LlavaForConditionalGeneration` 类中的成员变量 `self.multi_modal_projector`，并加入一个单层 Linear Layer 的 `self.mm_projector`（取这个名字这也是为了和 checkpoints 中的名称对应）：

```python
# llavar.py

from vllm.model_executor.models.llava import LlavaForConditionalGeneration
class LLavaRForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(...) -> None
    super().__init__(...)

    delattr(self, "multi_modal_projector")
    self.mm_projector = nn.Linear(
        in_features=config.vision_config.hidden_size,
        out_features=config.text_config.hidden_size,
        bias=True,
    )
```

至此，模型结构已经修改至与 `LLaVAR` 模型一致。

## 重写 `load_weights` 函数逻辑

`load_weights` 函数是 `vllm` 中所有模型都具有的一个成员函数，会在 `vllm.LLM` 初始化时被调用，用于将加载 checkpoints 中的参数加载到模型中。具体在 [`ModelLoader`](https://github.com/vllm-project/vllm/blob/0f0d8bc065f3608e7657a9696f5d2d7c0d6722d1/vllm/model_executor/model_loader/loader.py#L264) 被调用。该函数接受一个类似于 `state_dict` 的迭代器参数，迭代器中每一项为 (参数名，Tensor) 的二元组。重写该函数的目标就是将这些参数加载到模型中，这一步的逻辑，以及最简单的实际情况就是拿到一个 weights，然后找到对应的 parameter，然后 copy 就完事了。

`LlavaForConditionalGeneration` 类实现的 [`load_weights`](https://github.com/vllm-project/vllm/blob/0f0d8bc065f3608e7657a9696f5d2d7c0d6722d1/vllm/model_executor/models/llava.py#L306) 逻辑较为复杂，原因主要有以下两点：

1. `llava-1.5-7b-hf` checkpoints 中的参数名与 `vllm` 中实现的 `LlavaForConditionalGeneration` 类不对应。比如出于优化的目的，`LlavaForConditionalGeneration` 将 `llava-1.5-7b-hf` 中 LLaMA self-attention 部分的 `q_proj`, `k_proj`, `v_proj` 三个 Linear Layer 合并为了一个 `qkv_proj`，以及将 LLaMA MLP 部分的 `gate_proj` 和 `up_proj` 合并为了 `gate_up_proj`，因此需要分多次将这些分散的参数加载到完整的 `qkv_proj` 和 `gate_up_proj` 中。
2. 还是出于优化的目的，`QKVParallelLinear`, `VocabParallelEmbedding` 等部分是面向 tensor parallelism 实现的，因此每一个 worker (或 GPU) 都只会 load checkpoints 中完整参数的一部分。不过好在这种部分加载 `vllm` 已经帮我们实现好了。`QKVParallelLinear` 和 `VocabParallelEmbedding` 等模块都有一个 `weight_loader` 函数，就是用于部分加载参数，以支持 tensor parallelism，如 [`QKVParallelLinear.weight_loader`](https://github.com/vllm-project/vllm/blob/0f0d8bc065f3608e7657a9696f5d2d7c0d6722d1/vllm/model_executor/layers/linear.py#L548)。在重写 `load_weights` 时，绝大部分情况可以直接调用 `vllm` 已实现的 `weight_loader` 函数，但是也需要对 `weight_loader` 函数接口足够理解。

落到 `LLaVAR` 的 `load_weights` 具体实现，首先需要知道 checkpoints 中和 `LlavaRForConditionalGeneration` 类中参数名的区别：

`LlavaRForConditionalGeneration` 类中的参数结构：

```python
- vision_tower
- mm_projector      # 刚刚修改的
-- mm_projector.weight
-- mm_projector.bias
- language_model
-- language_model.embed_tokens
-- language_model.norm
-- language_model.layers.0.self_attn.qkv_proj.weight
-- language_model.layers.0.self_attn.o_proj.weight
-- language_model.layers.0.self_attn.input_layernorm.weight
-- language_model.layers.0.self_attn.post_attention_layernorm.weight
-- language_model.layers.0.mlp.gate_up_proj.weight
-- language_model.layers.0.mlp.down_proj.weight
-- ......
- lm_head
-- lm_head.weight
```

`LLaVAR` 提供的 checkpoints 参数：

```python
- lm_head.weight
- model.norm.weight
- model.mm_projector.weight
- model.mm_projector.bias
- model.embed_tokens.weight
- model.layers.0.input_layernorm.weight
- model.layers.0.mlp.down_proj.weight
- model.layers.0.mlp.gate_proj.weight
- model.layers.0.mlp.up_proj.weight
- model.layers.0.post_attention_layernorm.weight
- model.layers.0.self_attn.q_proj.weight
- model.layers.0.self_attn.k_proj.weight
- model.layers.0.self_attn.v_proj.weight
- model.layers.0.self_attn.o_proj.weight
- model.layers.0.self_attn.rotary_emb.inv_freq
......
```

可以观察到，需要在 `load_weights` 函数中修改三个部分，使得 weights 与 parameters 对齐：

1. `model.mm_projector` 需要改名为 `mm_projector`，去掉前缀 `model.`；
2. 除了 `mm_projector` 意外，所有以 `model.` 开头的参数名都得改为 `language_model.`；
3. 手动加载 `vision_tower`。因为 `LLaVAR` 提供的 checkpoints 中并没有 `vision_tower` 的参数，但我们知道 `LLaVAR` 的 `vision_tower` 其实就是 `openai/clip-vit-large-patch14-336`，因此手动加载即可。

具体的实现基于 `LlavaForConditionalGeneration` 类的实现做一些修改即可：

```python
# llavar.py

from vllm.model_executor.models.llava import LlavaForConditionalGeneration
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_KEYS_TO_MODIFY_MAPPING = {
    # Prioritize replacing "model.mm_projector" with "mm_projector" rather than language_model.mm_projector
    # The earlier the position in the mapping dictionary, the higher the priority
    "model.mm_projector": "mm_projector",
    "model.": "language_model.",
}


class LLavaRForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(...) -> None:
        ...

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = False
            if "vision" in name:
                if self.vision_tower is not None:
                    # We only do sharding for language model and
                    # not vision model for now.
                    use_default_weight_loading = True
            else:
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    use_default_weight_loading = True
            if use_default_weight_loading:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
```

可以看到主要就是改了 `_KEYS_TO_MODIFY_MAPPING` 变量，其他保持一致就完成了。
有个小细节，在 `load_weights` 函数里我并没有加载 `vision_tower` 的参数，这是因为我发现这么做之后执行时会报错，`vision_tower` 参数与数据不在同一个设备上，也就是说 `vision_tower` 还在 CPU。我猜测是因为 `load_weights` 函数已经是在模型被 shard 之后，各个 GPU 在执行并将参数加载到自己的显存中的时候。因此这个时候加载 `vision_tower` 如不指定 `device` 则会加载到 CPU，若指定 `cuda` 也不知道该加载到哪个 GPU，因为 `vision_tower` 并没有实现 tensor parallelism，只会加载到一个 GPU 中。因此这个时候就应该直接在 `__init__` 函数中手动加载 `vision_tower` 的参数，然后再让 `vllm` 去决定加载到某个 GPU 中。因此完整的代码应当如下：

```python
# llavar.py

from vllm.model_executor.models.llava import LlavaForConditionalGeneration
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_KEYS_TO_MODIFY_MAPPING = {
    # Prioritize replacing "model.mm_projector" with "mm_projector" rather than language_model.mm_projector
    # The earlier the position in the mapping dictionary, the higher the priority
    "model.mm_projector": "mm_projector",
    "model.": "language_model.",
}


class LLavaRForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(
        self,
        config: LlavaConfig,
        vision_language_config: VisionLanguageConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None
    ) -> None:
        super().__init__(config, vision_language_config, cache_config, quant_config)

    delattr(self, "multi_modal_projector")
    self.mm_projector = nn.Linear(
        in_features=config.vision_config.hidden_size,
        out_features=config.text_config.hidden_size,
        bias=True,
    )
    self.vision_tower = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        torch_dtype=torch.float16
    )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = False
            if "vision" in name:
                if self.vision_tower is not None:
                    # We only do sharding for language model and
                    # not vision model for now.
                    use_default_weight_loading = True
            else:
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    use_default_weight_loading = True
            if use_default_weight_loading:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
```

## 注册 `LlavaRForConditionalGeneration` 模型

官方文档的最后一步就是注册自定义模型。由于 `vllm.LLM` 接受的 `model` 参数只是一个 `str` 类型，因此 `vllm` 是仅接受一个模型名称或路径，然后再去内部实例化。因此为了能够让 `vllm` 知道自定义模型的存在，就需要手动去注册模型。注册很简单，直接照着官方文档抄就可以了：

```python
from vllm import ModelRegistry
from llavar import LlavaRForConditionalGeneration
ModelRegistry.register_model("LlavaRForConditionalGeneration", LlavaRForConditionalGeneration)
```

不过在真正运行时，还需要为 `LlavaRForConditionalGeneration` 类注册几个装饰器。如原始的 [`LlavaForConditionalGeneration`](https://github.com/vllm-project/vllm/blob/0f0d8bc065f3608e7657a9696f5d2d7c0d6722d1/vllm/model_executor/models/llava.py#L90) 类就注册了 3 个装饰器，分别是：

- 增加多模态**特征**输入(`@MULTIMODAL_REGISTRY.register_image_feature_input()`)的方法，
- 增加多模态**像素**输入(`@MULTIMODAL_REGISTRY.register_image_pixel_input()`)的方法，
- 以及一个给出输入示例(`@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)`)的方法。

我并没有仔细的去查看这些装饰器方法的作用，但是通过在其他部分的单步调试我猜测这 3 个方法的作用分别为：

- 为 `LlavaRForConditionalGeneration` 提供图片特征输入支持；
- 为 `LlavaRForConditionalGeneration` 提供图片像素输入支持；
- 在使用自定义模型生成之前，`vllm` 会先使用 `get_dummy_image_data` 生成一批样本数据走一遍（可能用于收集一些显存使用信息？），然后才使用真实数据进行生成。

前两个装饰器应该是 LMM 必需注册其中一个的，否则应该就无法使用多模态数据；而最后一个应该是所有模型都需要注册的装饰器。

通过注册模型让 `vllm` 知道自定义模型的存在之后，还需要让 `vllm` 知道什么时候使用自定义模型。`vllm` 实例化模型是通过 [`ModelRegistry`](https://github.com/vllm-project/vllm/blob/0f0d8bc065f3608e7657a9696f5d2d7c0d6722d1/vllm/model_executor/model_loader/utils.py#L32) 实现的，通过在 `ModelRegistry` 里根据框架名字找的，所以框架名字就决定了 `vllm` 用什么模型，而框架名字又是来源于 [`model_config.hf_config`](https://github.com/vllm-project/vllm/blob/0f0d8bc065f3608e7657a9696f5d2d7c0d6722d1/vllm/model_executor/model_loader/utils.py#L23)。看变量名字很显然，`model_config.hf_config` 就是 `huggingface` 的 `config.json`，因此我们只需要将 `LLaVAR` checkpoint 文件夹下 `config.json` 的 `architectures` 字段从 `LlavaLlamaForCausalLM` 改为刚刚注册的 `LlavaRForConditionalGeneration` 就可以让 `vllm` 加载了。

实际上，`vllm` 中 `model_config.hf_config` 来源也的确就是 `huggingface` 的 `config.json`，可以参考：[link](https://github.com/vllm-project/vllm/blob/0f0d8bc065f3608e7657a9696f5d2d7c0d6722d1/vllm/config.py#L137)。

## LLaVAR Config 适配

如上所述，`vllm` 的 `model_config` 很大一部分来自于 `huggingface` 的 `config.json`，因此需要保证 `huggingface-transformers` 能够正确地加载 `config.json`。由于 LLaVAR 本质上是 LLaVA-1.3，是基于 LLaMA-1-13B 的 LMM，其开发时的 `transformers` 库版本较老 (4.28.0)，而 `vllm` 最新版本需要的 `transformers` 库的版本要求又非常新 (4.41.2)，因此会导致一些版本冲突，比如 `config.json` 文件一些字段发生了变化，这就需要先对 `config.json` 文件字段进行修正。

修正方法其实就是根据新版本 `transformers` 库的 `LlavaConfig` 的各个字段，对老版本的 `config.json` 修改、匹配就完事了。最后我将 `LLaVAR` 自带的 [`config.json`](https://huggingface.co/SALT-NLP/LLaVAR_delta/blob/main/config.json) 改成了如下：

```json
{
    "_name_or_path": "./llavar",
    "architectures": [
        "LlavaRForConditionalGeneration"
    ],
    "text_config": {
        "_name_or_path": "lmsys/vicuna-13b-v1.1",
        "architectures": [
            "LlamaForCausalLM"
        ],
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "rms_norm_eps": 1e-06,
        "torch_dtype": "float16",
        "vocab_size": 32003,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "intermediate_size": 13824,
        "pad_token_id": 0,
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "tie_word_embeddings": false
    },
    "vision_config": {
        "hidden_size": 1024,
        "image_size": 336,
        "intermediate_size": 4096,
        "model_type": "clip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "patch_size": 14,
        "projection_dim": 768
    },
    "freeze_mm_mlp_adapter": false,
    "initializer_range": 0.02,
    "max_position_embeddings": 2048,
    "max_sequence_length": 2048,
    "mm_hidden_size": 1024,
    "mm_use_im_start_end": true,
    "mm_vision_tower": "openai/clip-vit-large-patch14-336",
    "model_type": "llava",
    "sep_image_conv_front": false,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.28.0.dev0",
    "tune_mm_mlp_adapter": false,
    "use_cache": true,
    "use_mm_proj": true,
    "vision_feature_layer": -2,
    "vision_feature_select_strategy": "default"
}
```

值得一提的是，这个 `config.json` 中其实有非常多的字段并没有被 `vllm` 用上，比如 `mm_use_im_start_end`, `mm_hidden_size` 等等，实际上只需要修正一些必要字段如 `text_config` 和 `vision_config` 中的字段即可。`text_config` 等这些字段是必须被修正的，比如我就踩了坑：`text_config` 没有设置，导致 `LlamaConfig` 默认使用 7B 的配置，因此与 checkpoints 的 13B 不契合。


## 完结
至此就已经可以跟使用 `llava-1.5-7b-hf` 模型一样正常使用 `LLaVAR` 进行多卡并行推理了。

本文严格意义上仅涉及到了官方文档的第3、4步，第1、2步留作后面我适配 Qwen-VL 和 CogVLM2 再谈。

## 吐槽

本文虽然看起来短，但是是我疯狂试错快 10h 的结果，为此损失了一晚上 + 一下午的老头环 DLC 时间（急死我了）。其实说是 10h，实际上本文涉及到的所有改动只花费了 3h，剩下 7h 都在跟 checkpoint 搏斗...

因为遇到了以下两个认知方面的错误：

1. 之前看到 `LLaVA` 和 `LLaVAR` 都是 release **delta** 版本的参数，完全没在意。直到昨晚拿着 **delta** 版本的 checkpoints 加载模型，然后模型跑起来之后一直胡言乱语不说人话，对着 `vllm` 源码翻来覆去看 + google 了 3h 才反应过来这个 **delta** 版本是什么意思... 其实就是为了不违背 LLaMA 的 LICENCE，将最终模型的参数跟 LLaMA 模型参数做了减法得到的参数就是 release 出来的 **delta** 版本。然后因为这认知上的疏忽坑了我一晚上的老头环 DLC 时间。
2. 这位更是重量级。我将 **delta** 版本的参数恢复之后直接将原来的 `pytorch_model-00001-of-00003.bin` 等模型文件名加了个前缀 `delta-` (`delta-pytorch_model-00001-of-00003.bin`)，继续放在 checkpoints 文件夹下，看似没有问题。然而跑起来之后发现模型依然胡言乱语不说人话，但是用 [`LLaVAR` Github Repo](https://github.com/SALT-NLP/LLaVAR) 的代码却能够正常说话，还能有这么奇怪的事？于是又是 4h 翻来覆去看 `vllm` 源码 + google... 最终发现问题出在了 `delta-pytorch_model-00001-of-00003.bin` 上面。我之前看过 `huggingface` 的 `.from_pretrained` 方法的加载 checkpoints 的逻辑，是取 `pytorch_model.bin.index.json` 文件的所有 value 的并集作为加载对象，非常合理正确，然后我就以为 `vllm` 也是一样的。直到我看到了 [`vllm` 的加载逻辑](https://github.com/vllm-project/vllm/blob/0f0d8bc065f3608e7657a9696f5d2d7c0d6722d1/vllm/model_executor/model_loader/loader.py#L156)，它居然是将 checkpoints 文件夹下所有 `.bin` 文件都加载？？？也就是说之前的 `delta-pytorch_model-00001-of-00003.bin` 文件也被加载了，所以有一些正确的参数就被 `delta` 版本的参数覆盖了，从而导致模型一直在胡言乱语不说人话。然后我把 `delta` 文件全删了，模型推理终于正常了。此时我想着一整个被用于 debug 的假期下午，只想大喊：“vllm 我 \*\*\*\*，你 \*\*\*\*！”

![红温](../assets/img/mimes/hongwen.png)
