import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from PIL import Image
from peft import LoraConfig
from huggingface_hub import hf_hub_download
from diffusers import HiDreamImageTransformer2DModel
# from instruction_refinement import refine_instruction
from safetensors.torch import load_file
import argparse, os, pickle, math
from tqdm.auto import tqdm
import json

PROMPT = "In this image, change {target}. {edit_instruction}. Do not change anything else in the image."

def parse_args():
    parser = argparse.ArgumentParser("generate intelligent manipulations using VLM.")
    parser.add_argument("--image_folder", type=str, help="original images folder", default="../PIC_2.0/image/train/")
    parser.add_argument("--json_folder", type=str, help="path to suggested manipulations json folder", default="outputs/chatgpt-4o-latest_PIC_2.0")
    parser.add_argument("--current_chunk", default=0, type=int)
    parser.add_argument("--total_chunks", default=1, type=int)
    parser.add_argument("--output_dir", default="outputs", type=str)
    args = parser.parse_args()
    return args

args = parse_args()

# Set to True to enable instruction refinement and transformer model
ENABLE_REFINE = True

# Load models
tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

# Configure transformer model if refinement is enabled
transformer = None
reload_keys = None
if ENABLE_REFINE:
    transformer = HiDreamImageTransformer2DModel.from_pretrained("HiDream-ai/HiDream-I1-Full", subfolder="transformer")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["to_k", "to_q", "to_v", "to_out", "to_k_t", "to_q_t", "to_v_t", "to_out_t", "w1", "w2", "w3", "final_layer.linear"],
        init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)
    transformer.max_seq = 4608
    lora_ckpt_path = hf_hub_download(repo_id="HiDream-ai/HiDream-E1-Full", filename="HiDream-E1-Full.safetensors")
    lora_ckpt = load_file(lora_ckpt_path, device="cuda")
    src_state_dict = transformer.state_dict()
    reload_keys = [k for k in lora_ckpt if "lora" not in k]
    reload_keys = {
        "editing": {k: v for k, v in lora_ckpt.items() if k in reload_keys},
        "refine": {k: v for k, v in src_state_dict.items() if k in reload_keys},
    }
    info = transformer.load_state_dict(lora_ckpt, strict=False)
    assert len(info.unexpected_keys) == 0

# Initialize pipeline
if ENABLE_REFINE:
    pipe = HiDreamImageEditingPipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Full",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
        transformer=transformer,
    )
else:
    pipe = HiDreamImageEditingPipeline.from_pretrained(
        "HiDream-ai/HiDream-E1-Full",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    )

# Move pipeline to GPU
pipe = pipe.to("cuda", torch.bfloat16)

args.output_folder = f"{args.output_dir}/hidream_images/{os.path.basename(args.json_folder)}"
os.makedirs(f"{args.output_dir}", exist_ok=True)
os.makedirs(f"{args.output_dir}/hidream_images", exist_ok=True)
os.makedirs(args.output_folder, exist_ok=True)

error_file_path = args.output_folder+f"hidream_chunk{args.current_chunk}.pkl"
if "emodb" in args.image_folder:
    error_file_path = args.output_folder+f"hidream_emodb_chunk{args.current_chunk}.pkl"
elif "framesdb" in args.image_folder:
    error_file_path = args.output_folder+f"hidream_framesdb_chunk{args.current_chunk}.pkl"
elif "mscoco" in args.image_folder:
    error_file_path = args.output_folder+f"hidream_mscoco_chunk{args.current_chunk}.pkl"
elif "ade20k" in args.image_folder:
    error_file_path = args.output_folder+f"hidream_ade20k_chunk{args.current_chunk}.pkl"
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
            for effect_itr in range(len(suggested_manipulations_json["Effects"])):
                if (not os.path.isfile(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")) and (f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}" not in blocked_ims_list):
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
                    original_width, original_height = img.size
                    max_len = max(original_height, original_width)
                    if max_len > 768:
                        if max_len == original_height:
                            img = img.resize((round(original_width*768/original_height), round(original_height*768/original_height)))
                        else:
                            img = img.resize((round(original_width*768/original_width), round(original_height*768/original_width)))
                    # img = img.resize((768, 768))
                    # Generate image
                    image = pipe(
                        prompt=prmpt,
                        negative_prompt="low resolution, blur",
                        image=img,
                        guidance_scale=5.0,
                        image_guidance_scale=4.0,
                        num_inference_steps=28,
                        generator=torch.Generator("cuda").manual_seed(3),
                        refine_strength=0.3 if ENABLE_REFINE else 0.0,
                        reload_keys=reload_keys,
                    ).images[0]

                    # Resize back to original dimensions and save
                    image = image.resize((original_width, original_height))
                    image.save(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")
        except Exception as exc:
            print(exc, f"for file {im}")


# # Refine instruction if enabled
# refined_instruction = refine_instruction(src_image=test_image, src_instruction=instruction)
# print(f"Original instruction: {instruction}")
# print(f"Refined instruction: {refined_instruction}")