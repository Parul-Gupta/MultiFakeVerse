# Use the modified diffusers & peft library
import sys
import os, math
from tqdm.auto import tqdm
import json
# workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../icedit"))

# if workspace_dir not in sys.path:
#     sys.path.insert(0, workspace_dir)
    
from diffusers import FluxFillPipeline

# Below is the original library
import torch
from PIL import Image
import numpy as np
import argparse
import random

PROMPT = "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but, change {target}. {edit_instruction}. Do not change anything else in the image."

def parse_args():
    parser = argparse.ArgumentParser("generate intelligent manipulations using VLM.")
    parser.add_argument("--image_folder", type=str, help="original images folder", default="../PIC_2.0/image/train/")
    parser.add_argument("--json_folder", type=str, help="path to suggested manipulations json folder", default="outputs/chatgpt-4o-latest_PIC_2.0")
    parser.add_argument("--current_chunk", default=0, type=int)
    parser.add_argument("--total_chunks", default=1, type=int)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--flux-path", type=str, default='black-forest-labs/flux.1-fill-dev', help="Path to the model")
    parser.add_argument("--lora-path", type=str, default='RiverZ/normal-lora', help="Path to the LoRA weights")
    parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable CPU offloading for the model")
    args = parser.parse_args()
    return args

args = parse_args()
pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(args.lora_path)

if args.enable_model_cpu_offload:
    pipe.enable_model_cpu_offload() 
else:
    pipe = pipe.to("cuda")

args.output_folder = f"{args.output_dir}/icedit_images/{os.path.basename(args.json_folder)}"
gpt_folder = f"{args.output_dir}/gpt_images/{os.path.basename(args.json_folder)}"
os.makedirs(f"{args.output_dir}", exist_ok=True)
os.makedirs(f"{args.output_dir}/icedit_images", exist_ok=True)
os.makedirs(args.output_folder, exist_ok=True)

error_file_path = args.output_folder+f"icedit_chunk{args.current_chunk}.pkl"
if "emodb" in args.image_folder:
    error_file_path = args.output_folder+f"icedit_emodb_chunk{args.current_chunk}.pkl"
elif "framesdb" in args.image_folder:
    error_file_path = args.output_folder+f"icedit_framesdb_chunk{args.current_chunk}.pkl"
elif "mscoco" in args.image_folder:
    error_file_path = args.output_folder+f"icedit_mscoco_chunk{args.current_chunk}.pkl"
elif "ade20k" in args.image_folder:
    error_file_path = args.output_folder+f"icedit_ade20k_chunk{args.current_chunk}.pkl"
if os.path.exists(error_file_path):
    blocked_ims_list = pickle.load(open(error_file_path, "rb"))
else:
    blocked_ims_list = []

im_list = sorted(os.listdir(args.image_folder))
chunk_size = math.ceil(len(im_list)/args.total_chunks)

im_list = im_list[chunk_size*args.current_chunk : min(chunk_size*(args.current_chunk + 1), len(im_list))]
print("Current Chunk:", args.current_chunk)
print(f"Data slice: {chunk_size * args.current_chunk} : {min(chunk_size * (args.current_chunk + 1), len(im_list))}")

for im in tqdm(im_list):
    if os.path.exists(os.path.join(args.json_folder, f"{im}.json")) and os.path.getsize(os.path.join(args.json_folder, f"{im}.json")) != 0:
        try:
            suggested_manipulations_json = json.load(open(os.path.join(args.json_folder, f"{im}.json"), "r"))
            img = Image.open(os.path.join(args.image_folder, im))
            original_width, original_height = img.size
            for effect_itr in range(len(suggested_manipulations_json["Effects"])):
                if (os.path.isfile(f"{gpt_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")) and (not os.path.isfile(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")) and (f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}" not in blocked_ims_list):
                    curr_effect = suggested_manipulations_json["Effects"][effect_itr]
                    chg_tgt = curr_effect["Change Target"].lower()

                    #  fix issue with Explanation json
                    if len(curr_effect["Explanation"]) == 1 and "," in curr_effect["Explanation"][0]:
                        curr_effect["Explanation"] = curr_effect["Explanation"][0].split(",")
                        curr_effect["Explanation"][1] = curr_effect["Explanation"][1].replace('\'', '').strip()
                        print(im, curr_effect["Explanation"])

                    if "human" not in chg_tgt and "object" not in chg_tgt and "text" not in chg_tgt:
                        tgt = f"""{curr_effect["Change Target"]}, particularly {curr_effect["Explanation"][0]}"""
                    else:
                        if curr_effect["Explanation"][0].lower().startswith("the "):
                            tgt = curr_effect["Explanation"][0]
                        else:
                            tgt = f"""the {curr_effect["Explanation"][0]}"""

                    prmpt = PROMPT.format(target=tgt, edit_instruction=curr_effect["Explanation"][1])

                    if img.size[0] != 512:
                        print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
                        new_width = 512
                        scale = new_width / img.size[0]
                        new_height = int(img.size[1] * scale)
                        new_height = (new_height // 8) * 8  
                        img = img.resize((new_width, new_height))
                        print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")

                    width, height = img.size
                    combined_image = Image.new("RGB", (width * 2, height))
                    combined_image.paste(img, (0, 0))
                    combined_image.paste(img, (width, 0))
                    mask_array = np.zeros((height, width * 2), dtype=np.uint8)
                    mask_array[:, width:] = 255 
                    mask = Image.fromarray(mask_array)

                    result_image = pipe(
                        prompt=prmpt,
                        image=combined_image,
                        mask_image=mask,
                        height=height,
                        width=width * 2,
                        guidance_scale=50,
                        num_inference_steps=28,
                        generator=torch.Generator("cpu").manual_seed(args.seed) if args.seed is not None else None,
                    ).images[0]

                    result_image = result_image.crop((width,0,width*2,height))

                    # Resize back to original dimensions and save
                    result_image = result_image.resize((original_width, original_height))
                    result_image.save(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")

                    print(f"\033[92mResult saved as {args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}\033[0m")
        except Exception as exc:
            print(exc, f"for file {im}")