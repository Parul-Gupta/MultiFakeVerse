import os, argparse, json
from tqdm.auto import tqdm
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import pickle, math, sys
from openai import OpenAI
import base64
import random

from ratelimit import limits, sleep_and_retry

# 10 calls per minute
CALLS = 8
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def check_limit():
    ''' Empty function just to check for calls to API '''
    return

PROMPT = "In this image, change {target}. {edit_instruction}. Do not change anything else in the image."

def parse_args():
    parser = argparse.ArgumentParser("generate intelligent manipulations using VLM.")
    parser.add_argument("--model_version", type=str, choices=["gemini-2.0-flash-exp-image-generation", "gpt-image-1"], default="gemini-2.0-flash-exp-image-generation")
    parser.add_argument("--image_folder", type=str, help="original images folder", default="../PIC_2.0/image/train/")
    parser.add_argument("--json_folder", type=str, help="path to suggested manipulations json folder", default="outputs/chatgpt-4o-latest_PIC_2.0")
    parser.add_argument("--current_chunk", default=0, type=int)
    parser.add_argument("--total_chunks", default=1, type=int)
    parser.add_argument("--api_dict_key", type=str, default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
    parser.add_argument("--api_list_itr", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs", type=str)
    args = parser.parse_args()
    return args

def remove_duplicates(orig_effects, curr_effects):
    return

def generate_gpt_manipulationa(args):
    args.output_folder = f"{args.output_dir}/gpt_images/{os.path.basename(args.json_folder)}"
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/gpt_images", exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    error_file_path = args.output_folder+f"gpt_chunk{args.current_chunk}.pkl"
    if "emodb" in args.image_folder:
        error_file_path = args.output_folder+f"gpt_emodb_chunk{args.current_chunk}.pkl"
    elif "framesdb" in args.image_folder:
        error_file_path = args.output_folder+f"gpt_framesdb_chunk{args.current_chunk}.pkl"
    elif "mscoco" in args.image_folder:
        error_file_path = args.output_folder+f"gpt_mscoco_chunk{args.current_chunk}.pkl"
    elif "ade20k" in args.image_folder:
        error_file_path = args.output_folder+f"gpt_ade20k_chunk{args.current_chunk}.pkl"
    if os.path.exists(error_file_path):
        blocked_ims_list = pickle.load(open(error_file_path, "rb"))
    else:
        blocked_ims_list = []

    im_list = sorted(os.listdir(args.image_folder))
    chunk_size = math.ceil(len(im_list)/args.total_chunks)

    im_list = im_list[chunk_size*args.current_chunk : min(chunk_size*(args.current_chunk + 1), len(im_list))]
    print("Current Chunk:", args.current_chunk)
    print(f"Data slice: {chunk_size * args.current_chunk} : {min(chunk_size * (args.current_chunk + 1), len(im_list))}")

    # randomly sample 10% of the real images, rather than going serially
    im_list = random.sample(im_list, k=int(0.1*len(im_list)))
    print(f"New length = {len(im_list)}")

    client = OpenAI()

    for im in tqdm(im_list):
        if os.path.exists(os.path.join(args.json_folder, f"{im}.json")) and os.path.getsize(os.path.join(args.json_folder, f"{im}.json")) != 0:
            try:
                suggested_manipulations_json = json.load(open(os.path.join(args.json_folder, f"{im}.json"), "r"))
                img = open(os.path.join(args.image_folder, im), "rb")
                effect_itr_list = list(range(len(suggested_manipulations_json["Effects"])))
                alreadyPresent = False
                for effect_itr in effect_itr_list:
                    if os.path.isfile(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}"):
                        alreadyPresent = True
                        break
                
                if alreadyPresent:
                    continue
                iter_list = random.sample(effect_itr_list, 1)
                for effect_itr in iter_list:
                    if (not os.path.isfile(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")) and (f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}" not in blocked_ims_list):
                        try:
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
                            result = client.images.edit(
                                model=args.model_version,
                                image=[img],
                                prompt=prmpt,
                                quality="low"
                            )
                            if result is not None and result.data is not None and len(result.data)>0 and result.data[0].b64_json is not None:
                                image_base64 = result.data[0].b64_json
                                image_bytes = base64.b64decode(image_base64)

                                # Save the image to a file
                                with open(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}", "wb") as f:
                                    f.write(image_bytes)
                            else:
                                # content blocked probably
                                blocked_ims_list.append(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")
                                pickle.dump(blocked_ims_list, open(error_file_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

                        except Exception as excp:
                            print(excp, f"for file {im}")

            except Exception as exc:
                print(exc, f"for image {im}")

    return

def generate_gemini_manipulations(args):
    api_key_dict = {}

    api_key_list = api_key_dict[args.api_dict_key][args.api_list_itr:]

    if args.json_folder.endswith("/"):
        args.json_folder = args.json_folder[:-1]
    args.output_folder = f"{args.output_dir}/gemini_images/{os.path.basename(args.json_folder)}"
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/gemini_images", exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)
    error_file_path = args.output_folder+f"chunk{args.current_chunk}.pkl"
    if "emodb" in args.image_folder:
        error_file_path = args.output_folder+f"emodb_chunk{args.current_chunk}.pkl"
    elif "framesdb" in args.image_folder:
        error_file_path = args.output_folder+f"framesdb_chunk{args.current_chunk}.pkl"
    elif "mscoco" in args.image_folder:
        error_file_path = args.output_folder+f"mscoco_chunk{args.current_chunk}.pkl"
    elif "ade20k" in args.image_folder:
        error_file_path = args.output_folder+f"ade20k_chunk{args.current_chunk}.pkl"
    if os.path.exists(error_file_path):
        blocked_ims_list = pickle.load(open(error_file_path, "rb"))
    else:
        blocked_ims_list = []

    im_list = sorted(os.listdir(args.image_folder))
    chunk_size = math.ceil(len(im_list)/args.total_chunks)

    im_list = im_list[chunk_size*args.current_chunk : min(chunk_size*(args.current_chunk + 1), len(im_list))]
    print("Current Chunk:", args.current_chunk)
    print(f"Data slice: {chunk_size * args.current_chunk} : {min(chunk_size * (args.current_chunk + 1), len(im_list))}")

    api_itr = 0
    client = genai.Client(api_key=api_key_list[api_itr])
    print("api key = ", api_itr, api_key_list[api_itr])
    minute_limit_exceed = 0

    for im in tqdm(im_list):
        if os.path.exists(os.path.join(args.json_folder, f"{im}.json")) and os.path.getsize(os.path.join(args.json_folder, f"{im}.json")) != 0:
            try:
                suggested_manipulations_json = json.load(open(os.path.join(args.json_folder, f"{im}.json"), "r"))
                # if "chatgpt" not in args.json_folder and os.path.exists(os.path.join(args.json_folder.replace("gemini-2.0-flash", "chatgpt-4o-latest"), f"{im}.json")):
                #     # Do not perform manipulations that are already done through chatgpt suggestions
                #     orig_suggested_man_json = json.load(open(os.path.join(args.json_folder.replace("gemini-2.0-flash", "chatgpt-4o-latest"), f"{im}.json"), "r"))
                #     orig_effects = orig_suggested_man_json["Effects"]
                #     to_remove_orig = []
                #     for eff_itr in range(len(orig_effects)):
                #         if not os.path.exists(f"{args.output_folder.replace("gemini-2.0-flash", "chatgpt-4o-latest")}/{im[:-4]}_{eff_itr}{im[-4:]}"):
                #             to_remove_orig.append(eff_itr)
                #     orig_effects_short = [orig_effects[f_itr] for f_itr in range(len(orig_effects)) if f_itr not in to_remove_orig]
                #     to_remove_new = remove_duplicates(orig_effects_short, suggested_manipulations_json["Effects"])
                
                # else:
                #     to_remove_new = []
                img = Image.open(os.path.join(args.image_folder, im))
                for effect_itr in range(len(suggested_manipulations_json["Effects"])):
                    if (not os.path.isfile(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")) and (f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}" not in blocked_ims_list):
                        try:
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
                            check_limit()
                            print("Calling Image API...", im)
                            response = client.models.generate_content(
                                model=args.model_version,
                                contents=[prmpt, img],
                                config=types.GenerateContentConfig(
                                    response_modalities=['TEXT', 'IMAGE']
                                )
                            )

                            if response.candidates is not None and response.candidates[0].content is not None:
                                for part in response.candidates[0].content.parts:
                                    if part.text is not None:
                                        print(part.text)
                                    if part.inline_data is not None:
                                        image = Image.open(BytesIO(part.inline_data.data))
                                        image.save(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")
                            else:
                                # Content blocked probably
                                blocked_ims_list.append(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")
                                pickle.dump(blocked_ims_list, open(error_file_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

                        except Exception as exc:
                            print(exc, f"for file {im}")
                            if "GenerateRequestsPerDayPerProjectPerModel" in str(exc) or "'quota_limit_value': '0'" in str(exc) or "API_KEY_INVALID" in str(exc):
                                api_itr += 1
                                if api_itr >= len(api_key_list):
                                    args.api_dict_key = str(int(args.api_dict_key) + 1)
                                    if args.api_dict_key not in api_key_dict:
                                        print("Daily Quota exhausted...")
                                        sys.exit()
                                    api_key_list = api_key_dict[args.api_dict_key]
                                    api_itr = 0                                
                                client = genai.Client(api_key=api_key_list[api_itr])
                                print("api key = ", api_itr, api_key_list[api_itr])
                            elif "'quotaValue': '10'" in str(exc):
                                minute_limit_exceed += 1
                                if minute_limit_exceed == 3:
                                    minute_limit_exceed = 0
                                    api_itr += 1
                                    if api_itr >= len(api_key_list):
                                        args.api_dict_key = str(int(args.api_dict_key) + 1)
                                        if args.api_dict_key not in api_key_dict:
                                            print("Daily Quota exhausted...")
                                            sys.exit()
                                        api_key_list = api_key_dict[args.api_dict_key]
                                        api_itr = 0                                
                                    client = genai.Client(api_key=api_key_list[api_itr])
                                    print("api key = ", api_itr, api_key_list[api_itr])
                            # else:
                            #     blocked_ims_list.append(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")
                            #     pickle.dump(blocked_ims_list, open(args.output_folder+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
                            continue
            except Exception as excp:
                print(excp, im)
    return

def main():
    args = parse_args()
    if args.model_version == "gpt-image-1":
        generate_gpt_manipulationa(args)
    else:
        generate_gemini_manipulations(args)
    return

if __name__ == "__main__":
    main()
