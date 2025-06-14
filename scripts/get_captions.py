import argparse, os
import json

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-name", type=str,
                        default="Lin-Chen/ShareCaptioner")
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--image_folder", type=str, default="",
                        help="path to folder containing images to be captioned")
    parser.add_argument("--output_path", type=str, default="captions.json",
                        help="path to output folder, where captions json will be saved")
    parser.add_argument("--current_chunk", type=int, default=0)
    parser.add_argument("--total_chunks", type=int, default=1)
    args = parser.parse_args()
    return args


def auto_configure_device_map(num_gpus):
    # visual_encoder 算4层
    # internlm_model.model.embed_tokens 占用1层
    # norm 和 lm_head 占用1层
    # transformer.layers 占用 32 层
    # 总共34层分配到num_gpus张卡上
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus

    device_map = {
        'visual_encoder': 0,
        'ln_vision': 0,
        'Qformer': 0,
        'internlm_model.model.embed_tokens': 0,
        'internlm_model.model.norm': 0,
        'internlm_model.lm_head': 0,
        'query_tokens': 0,
        'flag_image_start': 0,
        'flag_image_end': 0,
        'internlm_proj': 0,
    }

    used = 6
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'internlm_model.model.layers.{i}'] = gpu_target
        used += 1

    return device_map


if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="cpu", trust_remote_code=True).eval().half()

    if args.num_gpus > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(args.num_gpus)
        model = dispatch_model(model, device_map=device_map)
    else:
        model.cuda()

    model.tokenizer = tokenizer

    image_list = sorted(os.listdir(args.image_folder))
    os.makedirs(args.output_path, exist_ok=True)

    if args.total_chunks > 1:
        chunk_str = f"_chunk_{args.current_chunk}"
        chunk_size = math.ceil(len(image_list)/args.total_chunks)
        image_list = image_list[chunk_size*args.current_chunk : min(chunk_size*(args.current_chunk + 1), len(image_list))]
        print("Current Chunk:", args.current_chunk)
        print(f"Data slice: {chunk_size * args.current_chunk} : {min(chunk_size * (args.current_chunk + 1), len(image_list))}")
    else:
        chunk_str = ""

    if "chatgpt" in args.image_folder:
        out_file_name = f"{args.output_path}/chatgpt_"
    elif "gemini-2.0-flash" in args.image_folder:
        out_file_name = f"{args.output_path}/gemini_"
    else:
        out_file_name = f"{args.output_path}/real_"
    
    if "PISC" in args.image_folder:
        out_file_name += "PISC"
    elif "PIPA" in args.image_folder:
        out_file_name += "PIPA"
    elif "emotic" in args.image_folder:
        out_file_name += "emotic"
    elif "PIC" in args.image_folder:
        out_file_name += "PIC_2.0"
    
    out_file_name += f"{chunk_str}.json"

    if os.path.exists(out_file_name):
        captions_dict = json.load(open(out_file_name, "r"))
        alreadyDoneIms = list(captions_dict.keys())
    else:
        captions_dict = {}
        alreadyDoneIms = []
    
    image_list = list(set(image_list).difference(set(alreadyDoneIms)))

    part_len = len(image_list)

    seg1 = '<|User|>:'
    seg2 = f'Analyze the image in a comprehensive and detailed manner.{model.eoh}\n<|Bot|>:'
    seg_emb1 = model.encode_text(seg1, add_special_tokens=True)
    seg_emb2 = model.encode_text(seg2, add_special_tokens=False)

    chunk_size = len(image_list)//args.batch_size

    if len(image_list) % args.batch_size != 0:
        chunk_size += 1

    for i in range(chunk_size):
        print(f'{i}/{chunk_size}')
        subs = []
        for j in range(args.batch_size):
            if i*args.batch_size+j < len(image_list):
                img_name = image_list[i*args.batch_size+j]
                img_path = f"{args.image_folder}/{img_name}"
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as exc:
                    print(exc, img_path)
                    continue
                subs.append(model.vis_processor(image).unsqueeze(0))
        if len(subs) == 0:
            break
        subs = torch.cat(subs, dim=0).cuda()
        tmp_bs = subs.shape[0]
        tmp_seg_emb1 = seg_emb1.repeat(tmp_bs, 1, 1)
        tmp_seg_emb2 = seg_emb2.repeat(tmp_bs, 1, 1)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                subs = model.encode_img(subs)
                input_emb = torch.cat(
                    [tmp_seg_emb1, subs, tmp_seg_emb2], dim=1)

                out_embeds = model.internlm_model.generate(inputs_embeds=input_emb,
                                                           max_length=500,
                                                           num_beams=3,
                                                           min_length=1,
                                                           do_sample=True,
                                                           repetition_penalty=1.5,
                                                           length_penalty=1.0,
                                                           temperature=1.,
                                                           eos_token_id=model.tokenizer.eos_token_id,
                                                           num_return_sequences=1,
                                                           )
        for j, out in enumerate(out_embeds):
            out[out == -1] = 2
            response = model.decode_text([out])
            # print(response)
            captions_dict[image_list[i*args.batch_size+j]] = response
        
        if i%16 == 0:
            with open(out_file_name, 'w') as f:
                json.dump(captions_dict, f, indent=4)

    with open(out_file_name, 'w') as f:
        json.dump(captions_dict, f, indent=4)
    print('Done')