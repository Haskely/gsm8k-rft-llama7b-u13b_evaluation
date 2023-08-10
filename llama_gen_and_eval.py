import json
import re
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


def main(
    gsm8k_test_jsonl: str = "gsm8k_test.jsonl",
    model_path: str = "OFA-Sys/gsm8k-rft-llama7b-u13b",
    is_bf16: bool = False,
    batch_size: int = 32,
    save_dir: str | None = None,
):
    print(f"main start, is_bf16:{is_bf16}, batch_size:{batch_size}")
    with open(gsm8k_test_jsonl, "r") as f:
        gsm8k_datas = [json.loads(line) for line in f]

    model, tokenizer = get_model(model_path, is_bf16=is_bf16)
    print("model loaded")

    batch_llama = get_batch_llama(model, tokenizer)

    if save_dir is None:
        save_dir = f"./output_{model.dtype}_bs{batch_size}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    gen_datas_jsonl = Path(save_dir) / "gen_datas.jsonl"
    start_index = (
        len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
    )
    print(f"start_index: {start_index}")
    for i in tqdm(range(start_index, len(gsm8k_datas), batch_size)):
        cur_gsm8k_batch = gsm8k_datas[i : i + batch_size]
        input_str_list, output_str_list = gsm8k_batch_gen(
            [d["question"] for d in cur_gsm8k_batch], batch_llama
        )
        for j, (gsm8k_data, input_str, output_str) in enumerate(
            zip(cur_gsm8k_batch, input_str_list, output_str_list)
        ):
            with open(gen_datas_jsonl, "a") as f:
                json.dump(
                    dict(
                        index=i + j,
                        gsm8k_data=gsm8k_data,
                        input_str=input_str,
                        output_str=output_str,
                    ),
                    f,
                )
                f.write("\n")

    # calculate acc
    with open(gen_datas_jsonl) as f:
        gen_datas = [json.loads(line) for line in f]

    correct_results = []
    wrong_results = []
    for gen in gen_datas:
        result = dict(
            **gen,
            extract_true_num=extract_last_num(gen["gsm8k_data"]["answer"]),
            extract_pred_num=extract_last_num(gen["output_str"]),
            is_correct=None,
        )
        if abs(result["extract_true_num"] - result["extract_pred_num"]) < 1e-3:
            result["is_correct"] = True
            correct_results.append(result)
        else:
            result["is_correct"] = False
            wrong_results.append(result)

    with open(Path(save_dir) / "correct.json", "w") as f:
        json.dump(correct_results, f, ensure_ascii=False, indent=4)
    with open(Path(save_dir) / "wrong.json", "w") as f:
        json.dump(wrong_results, f, ensure_ascii=False, indent=4)

    result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results)/(len(correct_results) + len(wrong_results))}"
    print(result)


def gsm8k_batch_gen(
    gsm8k_questions: list[str], batch_llm: Callable[[list[str]], list[str]]
):
    prompt_no_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
    )
    input_str_list = [prompt_no_input.format(query=q) for q in gsm8k_questions]
    output_str_list = batch_llm(input_str_list)
    return input_str_list, output_str_list


def get_batch_llama(model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
    @torch.inference_mode()
    def batch_llama(input_strs: list[str]) -> list[str]:
        input_ids_w_attnmask = tokenizer(
            input_strs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(
            input_ids=input_ids_w_attnmask.input_ids,
            attention_mask=input_ids_w_attnmask.attention_mask,
            generation_config=GenerationConfig(
                max_length=512,
                do_sample=False,
                temperature=0.0,  # t=0.0 raise error if do_sample=True
            ),
        ).tolist()
        real_output_ids = [
            output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)
        ]
        output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
        return output_strs

    return batch_llama


def get_model(model_path: str, is_bf16: bool = False):
    tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
    print(tokenizer.pad_token)
    print(tokenizer.bos_token)
    print(tokenizer.unk_token)
    print(tokenizer.eos_token)
    print(tokenizer.truncation_side)
    print(tokenizer.padding_side)

    if is_bf16:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
        ).cuda()
    model.eval()
    print(model.dtype)

    return model, tokenizer


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return None


if __name__ == "__main__":
    import fire

    fire.Fire(main)
