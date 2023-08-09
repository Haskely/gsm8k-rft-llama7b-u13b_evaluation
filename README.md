
尝试对 [OFA-Sys/gsm8k-rft-llama7b-u13b](https://huggingface.co/OFA-Sys/gsm8k-rft-llama7b-u13b) 进行 GSM8K 测分。

[gsm8k_test.jsonl](./gsm8k_test.jsonl) 就是官方的 https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl 测试集文件。


```
CUDA_VISIBLE_DEVICES=0 python llama_gen_and_eval.py
```
一张 A800-80G:
- dtype=bf16, batch_size=64 约 10min 跑完 `Accuracy=376/(376+943)=0.2850644427596664`
- dtype=bf16, batch_size=32 约 12min 跑完 `Accuracy=387/(387+932)=0.2934040940106141`
- dtype=fp32, batch_size=32 约 20min 跑完 `Accuracy=405/(405+914)=0.3070507960576194`

相关 issue: https://github.com/OFA-Sys/gsm8k-ScRel/issues/8