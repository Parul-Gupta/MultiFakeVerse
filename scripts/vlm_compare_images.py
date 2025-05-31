import os, argparse, json
from tqdm.auto import tqdm
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import pickle, math, sys
import glob

from ratelimit import limits, sleep_and_retry

# 14 calls per minute
CALLS = 13
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def check_limit():
    ''' Empty function just to check for calls to API '''
    return

PROMPT = """Compare these two images: the first is the original (Image A) and the second is its manipulated version (Image B). For each of the following categories, describe the changes observed, if any, and their potential perceptual impact on a human viewer. Finally, classify the changes in Image B into person-level, person-object level or person-scene level, depending upon whether a person has been changed, an object connected to the person has been changed or a component of the scene away from the person has been changed respectively. Note that a manipulated image can have multiple types of changes, so output multiple labels for such image. Strictly use the following structure in your response, and do not add anything else:
1. Emotion/Affect:
- Image A: [Describe emotion, mood, facial expression, atmosphere]
- Image B: [Describe changes]
- Perceptual Impact: [How might this affect viewer emotion or interpretation?]

2. Identity & Character Perception:
- Image A: [Describe apparent age, gender, trustworthiness, etc.]
- Image B: [Describe changes]
- Perceptual Impact: [Does this shift how the person is perceived?]

3. Social Signals & Status:
- Image A: [Describe clothing, posture, social roles, proximity]
- Image B: [Describe changes]
- Perceptual Impact: [Do power dynamics or relationships change?]

4. Scene Context & Narrative:
- Image A: [Describe implied story or setting]
- Image B: [Describe changes]
- Perceptual Impact: [Does the story or situation change?]

5. Manipulation Intent:
- Description: [What might be the intent behind the manipulation?]
- Perceptual Impact: [Does the edit appear deceptive, persuasive, aesthetic, etc.?]

6. Ethical Implications:
- Description: [Could the manipulation mislead or cause harm?]
- Assessment: [Mild / Moderate / Severe ethical concern]

7. Type of changes:
 - [Person level / Person-Object level / Person-Scene level]
"""

def parse_args():
    parser = argparse.ArgumentParser("generate intelligent manipulations using VLM.")
    parser.add_argument("--model_version", type=str, choices=["gemini-2.0-flash", "gemini-2.0-flash-lite"], default="gemini-2.0-flash")
    parser.add_argument("--real_image", type=str, help="original images folder", default="../PIC_2.0/image/train/")
    parser.add_argument("--fake_image", type=str, help="generated images folder", default="outputs/gemini_images/chatgpt-4o-latest_PIC_2.0/")
    parser.add_argument("--output_folder", type=str, help="path to output comparisons json folder", default="outputs/vlm_comparisons")
    parser.add_argument("--current_chunk", default=0, type=int)
    parser.add_argument("--total_chunks", default=1, type=int)
    parser.add_argument("--api_dict_key", type=str, default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    parser.add_argument("--api_list_itr", type=int, default=0)
    args = parser.parse_args()
    return args


def predict_gemini_comparisons(args):
    api_key_dict = {}
    api_key_list = api_key_dict[args.api_dict_key][args.api_list_itr:]

    if args.fake_image.endswith("/"):
        args.fake_image = args.fake_image[:-1]
    os.makedirs(args.output_folder, exist_ok=True)
    args.output_folder = f"{args.output_folder}/{os.path.basename(args.fake_image)}"
    os.makedirs(args.output_folder, exist_ok=True)

    real_im_list = glob.glob(f"{args.real_image}/**/*.jpg", recursive=True) + glob.glob(f"{args.real_image}/**/*.png", recursive=True)
    real_im_dict = {os.path.basename(im):im for im in real_im_list}

    fake_im_list = sorted(os.listdir(args.fake_image))
    chunk_size = math.ceil(len(fake_im_list)/args.total_chunks)

    fake_im_list = fake_im_list[chunk_size*args.current_chunk : min(chunk_size*(args.current_chunk + 1), len(fake_im_list))]
    print("Current Chunk:", args.current_chunk)
    print(f"Data slice: {chunk_size * args.current_chunk} : {min(chunk_size * (args.current_chunk + 1), len(fake_im_list))}")

    api_itr = 0
    while api_key_list[api_itr] in erroneous:
        api_itr += 1
    client = genai.Client(api_key=api_key_list[api_itr])
    print("api key = ", api_itr, api_key_list[api_itr])
    # minute_limit_exceed = 0

    for im in tqdm(fake_im_list):
        fake_img = Image.open(f"{args.fake_image}/{im}")
        real_img = Image.open(real_im_dict[im[:-6]+im[-4:]])
        if not os.path.isfile(f"{args.output_folder}/{im}.txt"):
            try:
                # print("Calling Image API...", im)
                check_limit()
                response = client.models.generate_content(
                    model=args.model_version,
                    contents=[real_img, fake_img, PROMPT]
                )

                # print(response.usage_metadata)

                with open(f"{args.output_folder}/{im}.txt", "w") as f:
                    f.write(response.text.strip())

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
                # elif "'quotaValue': '15'" in str(exc):
                #     minute_limit_exceed += 1
                #     if minute_limit_exceed == 3:
                #         minute_limit_exceed = 0
                #         api_itr += 1
                #         if api_itr >= len(api_key_list):
                #             args.api_dict_key = str(int(args.api_dict_key) + 1)
                #             if args.api_dict_key not in api_key_dict:
                #                 print("Daily Quota exhausted...")
                #                 sys.exit()
                #             api_key_list = api_key_dict[args.api_dict_key]
                #             api_itr = 0                                
                #         client = genai.Client(api_key=api_key_list[api_itr])
                #         print("api key = ", api_itr, api_key_list[api_itr])
                continue
    return

def main():
    args = parse_args()
    predict_gemini_comparisons(args)
    return

if __name__ == "__main__":
    main()
