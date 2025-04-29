import base64, os, json, argparse
from openai import OpenAI
from google import genai
from PIL import Image
from tqdm.auto import tqdm

os.environ["OPENAI_API_KEY"] = "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"
PROMPT = "Given the attached image, identify the most important person in this image, and suggest minimal modifications to the image to obtain each of the following effects:\n(1) the person you identified appears naive\n(2) the person you identified appears nonchalant\n(3) the person you identified appears proud\n(4) the person you identified appears remorseful\n(5) the person you identified appears inexperienced\n(6) some factual information depicted in the image changes\nThe possible change targets for the modifications are: the objects or text or humans in the image. When suggesting changes to text in the image, be specific about what text is to be replaced and what text should be added instead.\nGive output as a valid JSON string in the following format:{\n'Most Important Person':<referring expression for most important person>\n'Effects':[\n{'Effect':<effect>,\n'Change Target': <change target>,\n'Explanation': [<referring expression for the change target>, <edit instruction>]}\n]\n}.\n\nDo not include any other information in the response."

def parse_args():
    parser = argparse.ArgumentParser("Script to prompt VLM to suggest intelligent manipulations.")
    parser.add_argument("--model_version", type=str, choices=["chatgpt-4o-latest", "gpt-4.1", "gemini-2.0-flash"], default="chatgpt-4o-latest")
    parser.add_argument("--image_folder", type=str, default="../PIC_2.0/image/train/")
    args = parser.parse_args()
    return args

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to your image
# image_path = "data/images/test.jpg"
# model_version = "dall-e-2"
# im = Image.open(image_path)
# Convert the image to a BytesIO object
# im = im.convert("RGBA")
# im.putalpha(0)
# im.save("im_mod.png")

# response = client.images.edit(
#     model=model_version,
#     image=open("im_mod.png", "rb"),
#     mask=open("data/images/im_5_mod.png", "rb"),
#     prompt="In the given image, make the baby visibly crying.",
#     n=1,
#     size="512x512"
# )
# print(response.data[0].url)

def prompt_openai(args):
    client = OpenAI()
    os.makedirs("outputs", exist_ok=True)
    dset_type = ""
    if "PIC_2.0" in args.image_folder:
        dset_type = "PIC_2.0"
    elif "PISC" in args.image_folder:
        dset_type = "PISC"
    else:
        dset_type = "emotic"
    args.output_folder = f"outputs/{args.model_version}_{dset_type}"
    os.makedirs(args.output_folder, exist_ok=True)
    for im in tqdm(os.listdir(args.image_folder)):
        if not os.path.isfile(os.path.join(args.output_folder, f"{im}.json")):
            try:
                _ = Image.open(os.path.join(args.image_folder, im))
                # Getting the Base64 string
                base64_image = encode_image(os.path.join(args.image_folder, im))
                response = client.responses.create(
                    model = args.model_version,
                    input = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": PROMPT
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            ],
                        }
                    ],
                )
                with open(os.path.join(args.output_folder, f"{im}.json"), "w") as fp:
                    json.dump(json.loads(response.output_text.replace("```json", "").replace("```", "").strip()), fp, indent=4)
            except Exception as exc:
                print(exc, f"for file {im}")
                continue
    return

def prompt_gemini(args):
    client = genai.Client(api_key="AIzaSyDpQfi3AVsP2biawwuowb1pQ0PBeig_1EA")

    os.makedirs("outputs", exist_ok=True)
    dset_type = ""
    if "PIC_2.0" in args.image_folder:
        dset_type = "PIC_2.0"
    elif "PISC" in args.image_folder:
        dset_type = "PISC"
    else:
        dset_type = "emotic"
    args.output_folder = f"outputs/{args.model_version}_{dset_type}"
    os.makedirs(args.output_folder, exist_ok=True)
    for im in tqdm(os.listdir(args.image_folder)):
        if not os.path.isfile(os.path.join(args.output_folder, f"{im}.json")):
            try:
                img = Image.open(os.path.join(args.image_folder, im))
                response = client.models.generate_content(
                    model = args.model_version,
                    contents = [img, PROMPT],
                )

                with open(os.path.join(args.output_folder, f"{im}.json"), "w") as fp:
                    json.dump(json.loads(response.text.replace("```json", "").replace("```", "").strip()), fp, indent=4)
            except Exception as exc:
                print(exc, f"for file {im}")
                continue
    return

def main():
    args = parse_args()
    if "gpt" in args.model_version:
        prompt_openai(args)
    else:
        prompt_gemini(args)
    return

if __name__ == "__main__":
    main()