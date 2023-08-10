
尝试对 [OFA-Sys/gsm8k-rft-llama7b-u13b](https://huggingface.co/OFA-Sys/gsm8k-rft-llama7b-u13b) 进行 GSM8K 测分。

[gsm8k_test.jsonl](./gsm8k_test.jsonl) 就是官方的 https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl 测试集文件。


```
CUDA_VISIBLE_DEVICES=0 python llama_gen_and_eval.py
```
一张 A800-80G:
- dtype=bf16, batch_size=64 约 10min 跑完 `Accuracy=376/(376+943)=0.2850644427596664`
- dtype=bf16, batch_size=32 约 12min 跑完 `Accuracy=387/(387+932)=0.2934040940106141`
- dtype=fp32, batch_size=32 约 20min 跑完 `Accuracy=405/(405+914)=0.3070507960576194`

## 更新
- dtype=fp32, batch_size=1  约 1.5h  跑完 `Accuracy=648/(648+671)=0.49128127369219105`

看来原因是 tokenizer 的 `pad_token = "[PAD]"` 在推理时没有被真正的忽略，参与了上下文计算，干扰了模型输出。

但是看
```python
from transformers import LlamaTokenizer

model_path = "OFA-Sys/gsm8k-rft-llama7b-u13b"
tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
print(tokenizer.pad_token)
print(tokenizer.pad_token_id)
tokenizer(["hello", "hello world, are you ok?"], padding=True)
```
输出:
```
You are using the legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This means that tokens that come after special tokens will not be properly handled. We recommend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565
[PAD]
0
{'input_ids': [[0, 0, 0, 0, 0, 0, 2, 22172], [2, 22172, 3186, 29892, 526, 366, 3431, 29973]], 'attention_mask': [[0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]}
```
很正常， attention mask 就算不传 generate 函数 https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/generation/utils.py#L1324 也会自动处理好，未能搞清究竟哪里出了差错。

## 更新
commit: [dd8e71](https://github.com/Haskely/gsm8k-rft-llama7b-u13b_evaluation/commit/dd8e714e52bb5b64b35f1895a738503ffc4ac64f) 手动将 attention_mask 传入，开了 batch_size=32 也能拿到 Accuracy=649/(649+670)=0.4920394238059136 的分数了：[output_torch.float32_bs32 ](./output_torch.float32_bs32)

难道 https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/generation/utils.py#L1324 没生效？

## 更新
查明是因为使用了 GenerationConfig，导致 https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/generation/utils.py#L1264 没有加载模型的默认参数，进而 pad_token_id 和 eos_token_id 均为 None，进而 https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/generation/utils.py#L614 返回了全一 mask 导致的。
解决方案，不用 GenerationConfig，或手动传入 attention mask

相关 issue: https://github.com/OFA-Sys/gsm8k-ScRel/issues/8
