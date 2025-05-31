from PIL import Image
import torch, argparse
import os, json
from tqdm.auto import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"

PROMPT = "Given the attached image, identify the most important person in this image, and suggest minimal modifications to the image to obtain each of the following effects:\n(1) the person you identified appears naive\n(2) the person you identified appears nonchalant\n(3) the person you identified appears proud\n(4) the person you identified appears remorseful\n(5) the person you identified appears inexperienced\n(6) some factual information depicted in the image changes\nThe possible change targets for the modifications are: the objects or text or humans in the image. When suggesting changes to text in the image, be specific about what text is to be replaced and what text should be added instead.\nGive output as a valid JSON string in the following format:{\n'Most Important Person':<referring expression for most important person>\n'Effects':[\n{'Effect':<effect>,\n'Change Target': <change target>,\n'Explanation': [<referring expression for the change target>, <edit instruction>]}\n]\n}.\n\nDo not include any other information in the response."

def parse_args():
    parser = argparse.ArgumentParser("Script to prompt VLM to suggest intelligent manipulations.")
    parser.add_argument("--model_version", type=str, choices=["Qwen/Qwen2.5-VL-72B-Instruct"], default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--image_folder", type=str, default="data/PIC_2.0/image/train/")
    args = parser.parse_args()
    return args

def prompt_qwen(args, sys_prompt="You are a helpful assistant.", max_new_tokens=4096):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_version, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto", cache_dir="./huggingface_cache", load_in_8bit=True)
    processor = AutoProcessor.from_pretrained(args.model_version, cache_dir="./huggingface_cache")

    os.makedirs("outputs", exist_ok=True)
    dset_type = ""
    if "PIC_2.0" in args.image_folder:
        dset_type = "PIC_2.0"
    elif "PISC" in args.image_folder:
        dset_type = "PISC"
    else:
        dset_type = "emotic"
    args.output_folder = f"outputs/{args.model_version.split('/')[-1]}_{dset_type}"
    os.makedirs(args.output_folder, exist_ok=True)

    for im in tqdm(os.listdir(args.image_folder)):
        if not os.path.isfile(os.path.join(args.output_folder, f"{im}.json")):
            try:
                image = Image.open(os.path.join(args.image_folder, im))
                image_local_path = "file://" + os.path.join(args.image_folder, im)
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": [
                            {"type": "text", "text": PROMPT},
                            {"image": image_local_path},
                        ]
                    },
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                print("text:", text)
                # image_inputs, video_inputs = process_vision_info([messages])
                inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
                inputs = inputs.to(model.device)

                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                with open(os.path.join(args.output_folder, f"{im}.json"), "w") as fp:
                    json.dump(json.loads(output_text[0].replace("```json", "").replace("```", "").strip()), fp, indent=4)

            except Exception as exc:
                print(exc, f"for file {im}")
                continue
    return

def main():
    args = parse_args()
    prompt_qwen(args)
    return

if __name__ == "__main__":
    main()