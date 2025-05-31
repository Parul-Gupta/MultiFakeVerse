import os
import random
import json
from PIL import Image

folder_path = "outputs/generated_images/gemini_images"
generated_image_folders = ["chatgpt-4o-latest_emotic", "chatgpt-4o-latest_PIC_2.0", "chatgpt-4o-latest_PIPA", "chatgpt-4o-latest_PISC"]

extra_folders = [i.replace("chatgpt-4o-latest", "gemini-2.0-flash") for i in generated_image_folders]

# extra_folders = []

TRAIN_SIZE = 0.7
VAL_SIZE = 0.1

def split_train_val_test():
    train_im_list_dict = {}
    val_im_list_dict = {}
    test_im_list_dict = {}
    all_im_list_dict = {}

    num_fakes_per_real_dict = {}

    for fldr in generated_image_folders:
        random.seed(100)
        dset_name = '_'.join(fldr.split('_')[1:])
        num_fakes_per_real_dict[dset_name] = {}

        images = os.listdir(os.path.join(folder_path, fldr))
        images_ids = [i[:-6]+i[-4:] for i in images]

        for im in images:
            try:
                _ = Image.open(f"{folder_path}/{fldr}/{im}")
            except Exception as exc:
                print(folder_path, fldr, im, exc)
                continue
            im_id = im[:-6]+im[-4:]
            if im_id not in num_fakes_per_real_dict[dset_name]:
                num_fakes_per_real_dict[dset_name][im_id] = 1
            else:
                num_fakes_per_real_dict[dset_name][im_id] += 1
        # unique and sorted image ids
        images_ids = sorted(list(set(images_ids)))
        random.shuffle(images_ids)
        train_len = int(TRAIN_SIZE*len(images_ids))
        val_len = int(VAL_SIZE*len(images_ids))
        train_im_list_dict[dset_name] = images_ids[:train_len]
        val_im_list_dict[dset_name] = images_ids[train_len:(train_len+val_len)]
        test_im_list_dict[dset_name] = images_ids[(train_len+val_len):]

        all_im_list_dict[dset_name] = images_ids

    for fldr in extra_folders:
        if os.path.exists(os.path.join(folder_path, fldr)):
            dset_name = '_'.join(fldr.split('_')[1:])
            images = os.listdir(os.path.join(folder_path, fldr))
            images_ids = [i[:-6]+i[-4:] for i in images]

            for im in images:
                try:
                    _ = Image.open(f"{folder_path}/{fldr}/{im}")
                except Exception as exc:
                    print(folder_path, fldr, im, exc)
                    continue
                im_id = im[:-6]+im[-4:]
                if im_id not in num_fakes_per_real_dict[dset_name]:
                    num_fakes_per_real_dict[dset_name][im_id] = 1
                else:
                    num_fakes_per_real_dict[dset_name][im_id] += 1
            # unique and sorted image ids
            images_ids = sorted(list(set(images_ids)))
            for im in images_ids:
                if im not in all_im_list_dict[dset_name]:
                    all_im_list_dict[dset_name].append(im)
                    # add non-overlapping to train
                    train_im_list_dict[dset_name].append(im)
    
    for fldr in generated_image_folders+extra_folders:
        new_folder_path = folder_path.replace("gemini_images", "gpt_images")
        if os.path.exists(os.path.join(new_folder_path, fldr)):
            dset_name = '_'.join(fldr.split('_')[1:])
            images = os.listdir(os.path.join(new_folder_path, fldr))
            images_ids = [i[:-6]+i[-4:] for i in images]

            for im in images:
                try:
                    _ = Image.open(f"{new_folder_path}/{fldr}/{im}")
                except Exception as exc:
                    print(new_folder_path, fldr, im, exc)
                    continue
                im_id = im[:-6]+im[-4:]
                if im_id not in num_fakes_per_real_dict[dset_name]:
                    num_fakes_per_real_dict[dset_name][im_id] = 1
                else:
                    num_fakes_per_real_dict[dset_name][im_id] += 1
            # unique and sorted image ids
            images_ids = sorted(list(set(images_ids)))
            for im in images_ids:
                if im not in all_im_list_dict[dset_name]:
                    all_im_list_dict[dset_name].append(im)
                    # add non-overlapping to train
                    train_im_list_dict[dset_name].append(im)
    
    for fldr in generated_image_folders+extra_folders:
        new_folder_path = folder_path.replace("gemini_images", "icedit_images")
        if os.path.exists(os.path.join(new_folder_path, fldr)):
            dset_name = '_'.join(fldr.split('_')[1:])
            images = os.listdir(os.path.join(new_folder_path, fldr))
            images_ids = [i[:-6]+i[-4:] for i in images]

            for im in images:
                try:
                    _ = Image.open(f"{new_folder_path}/{fldr}/{im}")
                except Exception as exc:
                    print(new_folder_path, fldr, im, exc)
                    continue
                im_id = im[:-6]+im[-4:]
                if im_id not in num_fakes_per_real_dict[dset_name]:
                    num_fakes_per_real_dict[dset_name][im_id] = 1
                else:
                    num_fakes_per_real_dict[dset_name][im_id] += 1
            # unique and sorted image ids
            images_ids = sorted(list(set(images_ids)))
            for im in images_ids:
                if im not in all_im_list_dict[dset_name]:
                    all_im_list_dict[dset_name].append(im)
                    # add non-overlapping to train
                    train_im_list_dict[dset_name].append(im)
    
    return train_im_list_dict, val_im_list_dict, test_im_list_dict, num_fakes_per_real_dict

def count_images(train_im_list_dict, val_im_list_dict, test_im_list_dict, num_fakes_per_real_dict):
    # Real images numbers
    total_real_train = 0
    total_real_val = 0
    total_real_test = 0
    total_real = 0
    for dset in train_im_list_dict:
        print(f"{dset} Real images--\n\tTrain: {len(train_im_list_dict[dset])} Val: {len(val_im_list_dict[dset])} Test: {len(test_im_list_dict[dset])} Total: {len(train_im_list_dict[dset])+len(val_im_list_dict[dset])+len(test_im_list_dict[dset])}")
        total_real_train += len(train_im_list_dict[dset])
        total_real_val += len(val_im_list_dict[dset])
        total_real_test += len(test_im_list_dict[dset])
        total_real += len(train_im_list_dict[dset])+len(val_im_list_dict[dset])+len(test_im_list_dict[dset])
    
    print("Overall Real Images--")
    print(f"\tTrain: {total_real_train} Val: {total_real_val} Test: {total_real_test} Total: {total_real}")

    # Fake images numbers
    fake_train_dict = {}
    fake_val_dict = {}
    fake_test_dict = {}

    for dset in train_im_list_dict:
        fake_train_dict[dset] = 0
        fake_val_dict[dset] = 0
        fake_test_dict[dset] = 0
        for im in train_im_list_dict[dset]:
            fake_train_dict[dset] += num_fakes_per_real_dict[dset][im]
        
        for im in val_im_list_dict[dset]:
            fake_val_dict[dset] += num_fakes_per_real_dict[dset][im]
        
        for im in test_im_list_dict[dset]:
            fake_test_dict[dset] += num_fakes_per_real_dict[dset][im]
    
    total_fake_train = 0
    total_fake_val = 0
    total_fake_test = 0
    total_fake = 0
    
    for dset in train_im_list_dict:
        print(f"{dset} Fake images--\n\tTrain: {fake_train_dict[dset]} Val: {fake_val_dict[dset]} Test: {fake_test_dict[dset]} Total: {fake_train_dict[dset]+fake_val_dict[dset]+fake_test_dict[dset]}")
        total_fake_train += fake_train_dict[dset]
        total_fake_val += fake_val_dict[dset]
        total_fake_test += fake_test_dict[dset]
        total_fake += fake_train_dict[dset]+fake_val_dict[dset]+fake_test_dict[dset]
    
    print("Overall Fake Images--")
    print(f"\tTrain: {total_fake_train} Val: {total_fake_val} Test: {total_fake_test} Total: {total_fake}")
    return

train_im_list_dict, val_im_list_dict, test_im_list_dict, num_fakes_per_real_dict = split_train_val_test()

json.dump({"train": train_im_list_dict, "val": val_im_list_dict, "test": test_im_list_dict}, open("outputs/splits.json", "w"), indent=4)
count_images(train_im_list_dict, val_im_list_dict, test_im_list_dict, num_fakes_per_real_dict)


"""
'outputs/generated_images/gemini_images/chatgpt-4o-latest_PISC/19967_3.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/02002_2.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/02879_5.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/03634_5.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/04165_2.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/05879_3.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/10921_1.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/11888_4.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/13687_3.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/19778_1.jpg'
'outputs/generated_images/gemini_images/gemini-2.0-flash_PISC/24141_1.jpg'
'outputs/generated_images/gemini_images/chatgpt-4o-latest_PISC/17539_0.jpg'


import shutil
import glob
import json

fldr_list = ["emotic", "PIPA_dataset", "PISC_Dataset", "PIC_2.0"]

splits_info = json.load(open("splits1.json"))

for fldr in fldr_list:
    if fldr == "emotic":
        dset = "emotic"
    elif "PIPA" in fldr:
        dset = "PIPA"
    elif "PISC" in fldr:
        dset = "PISC"
    else:
        dset = "PIC_2.0"
    
    all_reals = splits_info["train"][dset] + splits_info["val"][dset] + splits_info["test"][dset]

    all_ims = glob.glob(f"real_data/{fldr}/**/*.jpg", recursive=True) + glob.glob(f"real_data/{fldr}/**/*.png", recursive=True)

    for im in all_ims:
        im_name = im.split("/")[-1]
        if im_name in all_reals:
            shutil.copy(im, f"real_data_sub/{fldr}/{im_name}")
"""