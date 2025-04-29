import tqdm, os, argparse, json
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

PROMPT = "In this image, change {target}. {edit_instruction}. Do not change anything else in the image."

def parse_args():
    parser = argparse.ArgumentParser("generate intelligent manipulations using VLM.")
    parser.add_argument("--model_version", type=str, choices=["gemini-2.0-flash-exp-image-generation"], default="gemini-2.0-flash-exp-image-generation")
    parser.add_argument("--image_folder", type=str, help="original images folder", default="../PIC_2.0/image/train/")
    parser.add_argument("--json_folder", type=str, help="path to suggested manipulations json folder", default="outputs/gemini-2.0-flash_PIC_2.0")
    args = parser.parse_args()
    return args

def generate_gemini_manipulations(args):
    client = genai.Client(api_key="AIzaSyDpQfi3AVsP2biawwuowb1pQ0PBeig_1EA")
    args.output_folder = f"outputs/gemini_images/{os.path.basename(args.json_folder)}"
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/gemini_images", exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)
    for im in tqdm(os.listdir(args.image_folder)):
        if os.path.exists(os.path.join(args.json_folder, f"{im}.json")):
            suggested_manipulations_json = json.load(open(os.path.join(args.json_folder, f"{im}.json"), "r"))
            img = Image.open(os.path.join(args.image_folder, im))
            for effect_itr in range(len(suggested_manipulations_json["Effects"])):
                if not os.path.isfile(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}"):
                    curr_effect = suggested_manipulations_json["Effects"][effect_itr]
                    chg_tgt = curr_effect["Change Target"].lower()
                    if "human" not in chg_tgt and "object" not in chg_tgt and "text" not in chg_tgt:
                        tgt = f"""{curr_effect["Change Target"]}, particularly {curr_effect["Explanation"][0]}"""
                    else:
                        if curr_effect["Explanation"][0].lower().startswith("the"):
                            tgt = curr_effect["Explanation"][0]
                        else:
                            tgt = f"""the {curr_effect["Explanation"][0]}"""

                    prmpt = PROMPT.format(target=tgt, edit_instruction=curr_effect["Explanation"][1])
                    response = client.models.generate_content(
                        model=args.model_version,
                        contents=[prmpt, img],
                        config=types.GenerateContentConfig(
                            response_modalities=['IMAGE']
                        )
                    )
                    print(type(response.candidates[0].content))
                    for part in response.candidates[0].content.parts:
                        if part.text is not None:
                            print(part.text)
                        elif part.inline_data is not None:
                            image = Image.open(BytesIO(part.inline_data.data))
                            image.save(f"{args.output_folder}/{im[:-4]}_{effect_itr}{im[-4:]}")
    return

def main():
    args = parse_args()
    generate_gemini_manipulations(args)
    return

if __name__ == "__main__":
    main()