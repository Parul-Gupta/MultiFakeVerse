import argparse, os
from tqdm.auto import tqdm
import random
import json
import math
import spacy
nlp = spacy.load('en_core_web_sm')

def parse_args():
    parser = argparse.ArgumentParser("analyze the real-fake comparison outputs")
    parser.add_argument("--comparison_folder", "-c", type=str, default="outputs/vlm_comparisons")
    args = parser.parse_args()
    return args

def read_comparisons(args):
    to_remove_files = []
    if os.path.isfile("outputs/analyzed_comparisons.json"):
        content_dict = json.load(open("outputs/analyzed_comparisons.json", "r"))
    else:
        content_dict = {}
    
    folder_list = os.listdir(args.comparison_folder)
    for fldr in tqdm(folder_list, desc="Reading comparison outputs"):
        file_list = os.listdir(f"{args.comparison_folder}/{fldr}")
        if fldr not in content_dict:
            content_dict[fldr] = {}
        else:
            alreadyDone = content_dict[fldr]
            file_list = [i for i in file_list if i.replace(".txt", "") not in alreadyDone]
        for fl in tqdm(file_list):
            try:
                lines = open(f"{args.comparison_folder}/{fldr}/{fl}", "r", encoding='latin1').readlines()
                content = "".join(lines)
                if content.strip() == "" or "Emotion/Affect" not in content:
                    # empty file created or error in response
                    # os.remove(f"{args.comparison_folder}/{fldr}/{fl}")
                    to_remove_files.append(f"{args.comparison_folder}/{fldr}/{fl}")
                    # print(f"Removed empty file: {fldr}/{fl}")
                else:
                    content_dict[fldr][fl.replace(".txt", "")] = content
            except Exception as exc:
                print(fldr, fl, exc)
        
    json.dump(content_dict, open("outputs/analyzed_comparisons.json", "w"))
    # print(to_remove_files)
    return content_dict

def get_type_statistics(args):
    content_dict = read_comparisons(args)
    type_dict = {
        "Person": 0,
        "Object": 0,
        "Scene": 0,
        "Person-Object": 0,
        "Person-Scene": 0,
        "Object-Scene": 0,
        "Person-Object-Scene": 0,
        "None": 0
    }

    type_im_list_dict = {
        "Person": [],
        "Object": [],
        "Scene": [],
        "Person-Object": [],
        "Person-Scene": [],
        "Object-Scene": [],
        "Person-Object-Scene": []
    }

    for fldr in content_dict:
        for fl, content in content_dict[fldr].items():
            content = content.replace("*", "")
            if "Type of changes:" not in content:
                print(content, fldr, fl)
                continue
            tp = content.split("Type of changes:")[-1].strip()
            tp = tp.split("\n")
            tp = [i.replace("[", "").replace("]", "").replace("- ", "").strip() for i in tp]
            for idx, i in enumerate(tp):
                if "scene" in i.lower():
                    tp[idx] = "scene-level"
                elif "object" in i.lower():
                    tp[idx] = "object-level"
                elif "person" in i.lower():
                    tp[idx] = "person-level"
            
            # remove duplicates
            tp = list(set(tp))

            if len(tp) == 1:
                # only one kind of change
                if "scene" in tp[0].lower():
                    type_dict["Scene"]+=1
                    type_im_list_dict["Scene"].append(f"{fldr}/{fl}")
                elif "object" in tp[0].lower():
                    type_dict["Object"]+=1
                    type_im_list_dict["Object"].append(f"{fldr}/{fl}")
                elif "person" in tp[0].lower():
                    type_dict["Person"]+=1
                    type_im_list_dict["Person"].append(f"{fldr}/{fl}")
                elif "no " in tp[0].lower() or "nil" in tp[0].lower() or "not applicable" in tp[0].lower() or "none" in tp[0].lower() or "n/a" in tp[0].lower() or "na" in tp[0].lower() or "no change" in tp[0].lower() or "no significant change" in tp[0].lower() or "no types of change" in tp[0].lower() or "no type of change" in tp[0].lower() or "cannot determine" in tp[0].lower() or "same" in tp[0].lower() or "identical" in tp[0].lower() or "not " in tp[0].lower() or tp[0]=="":
                    type_dict["None"] += 1
                else:
                    print(tp, fldr, fl)
            
            elif len(tp) == 2:
                # two kinds of changes
                if "no " in tp[0].lower() or "nil" in tp[0].lower() or "not applicable" in tp[0].lower() or "none" in tp[0].lower() or "n/a" in tp[0].lower() or "na" in tp[0].lower() or "no change" in tp[0].lower() or "no significant change" in tp[0].lower() or "no types of change" in tp[0].lower() or "no type of change" in tp[0].lower() or "no " in tp[1].lower() or "nil" in tp[1].lower() or "not applicable" in tp[1].lower() or "none" in tp[1].lower() or "n/a" in tp[1].lower() or "na" in tp[1].lower() or "no change" in tp[1].lower() or "no significant change" in tp[1].lower() or "no types of change" in tp[1].lower() or "no type of change" in tp[1].lower() or "cannot determine" in tp[0].lower() or "cannot determine" in tp[1].lower() or "same" in tp[0].lower() or "identical" in tp[0].lower() or "same" in tp[1].lower() or "identical" in tp[1].lower() or "not " in tp[0].lower() or "not " in tp[1].lower():
                    type_dict["None"] += 1
                    continue
                if "scene" in tp[0].lower() or "scene" in tp[1].lower():
                    if "object" in tp[0].lower() or "object" in tp[1].lower():
                        type_dict["Object-Scene"] += 1
                        type_im_list_dict["Object-Scene"].append(f"{fldr}/{fl}")
                    elif "person" in tp[0].lower() or "person" in tp[1].lower():
                        type_dict["Person-Scene"] += 1
                        type_im_list_dict["Person-Scene"].append(f"{fldr}/{fl}")
                    else:
                        print(tp, fldr, fl)
                
                elif "object" in tp[0].lower() or "object" in tp[1].lower():
                    if "person" in tp[0].lower() or "person" in tp[1].lower():
                        type_dict["Person-Object"] += 1
                        type_im_list_dict["Person-Object"].append(f"{fldr}/{fl}")
                    else:
                        print(tp, fldr, fl)
                else:
                    print(tp, fldr, fl)
            
            elif len(tp) == 3:
                # all three kinds of changes
                if "no " in tp[0].lower() or "nil" in tp[0].lower() or "not applicable" in tp[0].lower() or "none" in tp[0].lower() or "n/a" in tp[0].lower() or "na" in tp[0].lower() or "no change" in tp[0].lower() or "no significant change" in tp[0].lower() or "no types of change" in tp[0].lower() or "no type of change" in tp[0].lower() or "no " in tp[1].lower() or "nil" in tp[1].lower() or "not applicable" in tp[1].lower() or "none" in tp[1].lower() or "n/a" in tp[1].lower() or "na" in tp[1].lower() or "no change" in tp[1].lower() or "no significant change" in tp[1].lower() or "no types of change" in tp[1].lower() or "no type of change" in tp[1].lower() or "no " in tp[2].lower() or "nil" in tp[2].lower() or "not applicable" in tp[2].lower() or "none" in tp[2].lower() or "n/a" in tp[2].lower() or "na" in tp[2].lower() or "no change" in tp[2].lower() or "no significant change" in tp[2].lower() or "no types of change" in tp[2].lower() or "no type of change" in tp[2].lower() or "cannot determine" in tp[0].lower() or "cannot determine" in tp[1].lower() or "cannot determine" in tp[2].lower() or "same" in tp[0].lower() or "identical" in tp[0].lower() or "same" in tp[1].lower() or "identical" in tp[1].lower()  or "same" in tp[2].lower() or "identical" in tp[2].lower() or "not " in tp[0].lower() or "not " in tp[1].lower() or "not " in tp[2].lower():
                    type_dict["None"] += 1
                    continue
                if "scene" in tp[0].lower() or "scene" in tp[1].lower() or "scene" in tp[2].lower():
                    if "object" in tp[0].lower() or "object" in tp[1].lower() or "object" in tp[2].lower():
                        if "person" in tp[0].lower() or "person" in tp[1].lower() or "person" in tp[2].lower():
                            type_dict["Person-Object-Scene"] += 1
                            type_im_list_dict["Person-Object-Scene"].append(f"{fldr}/{fl}")
                        else:
                            print(tp, fldr, fl)
                    else:
                        print(tp, fldr, fl)
                else:
                    print(tp, fldr, fl)
            
            elif len(tp)==4 and ("no " in tp[0].lower() or "nil" in tp[0].lower() or "not applicable" in tp[0].lower() or "none" in tp[0].lower() or "n/a" in tp[0].lower() or "na" in tp[0].lower() or "no change" in tp[0].lower() or "no significant change" in tp[0].lower() or "no types of change" in tp[0].lower() or "no type of change" in tp[0].lower() or "no " in tp[1].lower() or "nil" in tp[1].lower() or "not applicable" in tp[1].lower() or "none" in tp[1].lower() or "n/a" in tp[1].lower() or "na" in tp[1].lower() or "no change" in tp[1].lower() or "no significant change" in tp[1].lower() or "no types of change" in tp[1].lower() or "no type of change" in tp[1].lower() or "no " in tp[2].lower() or "nil" in tp[2].lower() or "not applicable" in tp[2].lower() or "none" in tp[2].lower() or "n/a" in tp[2].lower() or "na" in tp[2].lower() or "no change" in tp[2].lower() or "no significant change" in tp[2].lower() or "no types of change" in tp[2].lower() or "no type of change" in tp[2].lower() or "cannot determine" in tp[0].lower() or "cannot determine" in tp[1].lower() or "cannot determine" in tp[2].lower() or "same" in tp[0].lower() or "identical" in tp[0].lower() or "same" in tp[1].lower() or "identical" in tp[1].lower()  or "same" in tp[2].lower() or "identical" in tp[2].lower() or "no " in tp[3].lower() or "nil" in tp[3].lower() or "not applicable" in tp[3].lower() or "none" in tp[3].lower() or "n/a" in tp[3].lower() or "na" in tp[3].lower() or "no change" in tp[3].lower() or "no significant change" in tp[3].lower() or "no types of change" in tp[3].lower() or "no type of change" in tp[3].lower() or "cannot determine" in tp[3].lower() or "same" in tp[3].lower() or "identical" in tp[3].lower() or "not " in tp[3].lower() or "not " in tp[0].lower() or "not " in tp[1].lower() or "not " in tp[2].lower()):
                type_dict["None"] += 1
            else:
                print(tp, fldr, fl)
    
    print(type_dict)
    total = sum(list(type_dict.values()))

    import seaborn as sns
    # import module
    from matplotlib_venn import venn3, venn3_circles
    from matplotlib import pyplot as plt

    plt.figure(figsize=(6, 5))
    # depict venn diagram
    out = venn3(subsets=(round(100*type_dict["Person"]/total, 2), round(100*type_dict["Object"]/total, 2), round(100*type_dict["Person-Object"]/total, 2), round(100*type_dict["Scene"]/total, 2), round(100*type_dict["Person-Scene"]/total, 2), round(100*type_dict["Object-Scene"]/total, 2), round(100*type_dict["Person-Object-Scene"]/total, 2)),
        set_labels=('Person', 'Object', 'Scene'), 
        set_colors=("orange", "blue", "red"), alpha=0.7)

    # outline of circle line style and width
    venn3_circles(subsets=(round(100*type_dict["Person"]/total, 2), round(100*type_dict["Object"]/total, 2), round(100*type_dict["Person-Object"]/total, 2), round(100*type_dict["Scene"]/total, 2), round(100*type_dict["Person-Scene"]/total, 2), round(100*type_dict["Object-Scene"]/total, 2), round(100*type_dict["Person-Object-Scene"]/total, 2)),
                linestyle="dashed", linewidth=2)
    
    for text in out.set_labels:
        text.set_fontsize(14)
    for text in out.subset_labels:
        text.set_fontsize(15)

    # title of the venn diagram
    plt.title("Normalized Distribution of\ndifferent levels of changes in MultiFakeVerse", fontsize=18)
    plt.savefig("plots/Venn_Norm_Distribution_levels.pdf")

    for k,v in type_im_list_dict.items():
        if len(v)>=10:
            print(k, random.sample(v, 10))
        else:
            print(k, v)

        print()

    return

def get_ethical_concern_statistics(args):
    content_dict = read_comparisons(args)
    ethics_dict = {
        "mild": 0,
        "moderate": 0,
        "severe": 0,
        "none": 0
    }
    ethics_im_list_dict = {
        "mild": [],
        "moderate": [],
        "severe": []
    }
    for fldr in content_dict:
        for fl, content in content_dict[fldr].items():
            content = content.replace("*", "")
            if "Ethical Implications" not in content:
                print(content, fldr, fl)
                continue
            concern = content.split("Ethical Implications")[-1]
            if "Assessment:" in concern:
                concern = concern.split("Assessment:")[1]
            if "7. " in concern:
                concern = concern.split("7. ")[0]

            if "severe" in concern.lower() and "not severe" not in concern.lower():
                ethics_dict['severe'] += 1
                ethics_im_list_dict['severe'].append(f"{fldr}/{fl}")
            elif "moderate" in concern.lower():
                ethics_dict['moderate'] += 1
                ethics_im_list_dict['moderate'].append(f"{fldr}/{fl}")
            elif "mild" in concern.lower() or "minimal" in concern.lower() or "low" in concern.lower() or "unknown" in concern.lower() or "negligible" in concern.lower():
                ethics_dict['mild'] += 1
                ethics_im_list_dict['mild'].append(f"{fldr}/{fl}")
            elif "n/a" in concern.lower() or "no " in concern.lower() or "not " in concern.lower() or "na" in concern.lower() or "n.a" in concern.lower() or "none" in concern.lower():
                ethics_dict["none"] += 1
            else:
                print(concern, fldr, fl)
    
    total = sum(list(ethics_dict.values()))
    for k in ethics_dict:
        print(k, 100*ethics_dict[k]/total)

    for k,v in ethics_im_list_dict.items():
        if len(v)>=10:
            print(k, random.sample(v, 10))
        else:
            print(k, v)

        print()
    return

def get_emotion_statistics(args):
    content_dict = read_comparisons(args)
    impact_list = []
    for fldr in content_dict:
        for fl, content in content_dict[fldr].items():
            content = content.replace("*", "")
            if "Emotion/Affect" not in content:
                print(content, fldr, fl)
                continue
            impact = content.split("Emotion/Affect:")[-1]
            if "2." in impact:
                impact = impact.split("2.")[0]
            if "Perceptual Impact" in impact:
                impact = impact.split("Perceptual Impact")[1]
            impact = impact.split("\n")[0]
            if impact.startswith(":"):
                impact = impact[1:]
            impact_list.append(impact)
    
    # import re
    # from wordcloud import STOPWORDS

    # text = ' '.join(impact_list)

    # text = re.sub(r'[^A-Za-z\s]', '', text)

    # text = text.lower()
    total_chunks = 50
    chunk_size = math.ceil(len(impact_list)/total_chunks)
    doc_list = []
    for chnk in tqdm(range(total_chunks)):
        curr_set = impact_list[chnk*chunk_size:min((chnk+1)*chunk_size, len(impact_list))]
        text = '\n'.join(curr_set)
        doc = nlp(text)
        doc_list.append(doc)

    # Generating a word cloud with the adjetives of the story
    full_words = ""
    for doc in tqdm(doc_list):
        words = ' '.join(
            [ 
            token.norm_ for token in doc
            if token.is_alpha and not token.like_num and not token.is_stop and
                not token.is_currency and token.pos_ in ['ADJ']
            ]
        )
        full_words = full_words + " " + words

    # stopwords = set(STOPWORDS)
    # text = ' '.join(word for word in text.split() if word not in stopwords)
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title("Emotional Impact Word Cloud")
    plt.show()
    return

def get_identity_change_statistics(args):
    content_dict = read_comparisons(args)
    impact_list = []
    for fldr in content_dict:
        for fl, content in content_dict[fldr].items():
            content = content.replace("*", "")
            if "Identity & Character" not in content:
                print(content, fldr, fl)
                continue
            impact = content.split("Identity & Character")[-1]
            if "3." in impact:
                impact = impact.split("3.")[0]
            if "Perceptual Impact" in impact:
                impact = impact.split("Perceptual Impact")[1]
            impact = impact.split("\n")[0]
            if impact.startswith(":"):
                impact = impact[1:]
            impact_list.append(impact)
    
    # import re
    # from wordcloud import STOPWORDS

    # text = ' '.join(impact_list)

    # text = re.sub(r'[^A-Za-z\s]', '', text)

    # text = text.lower()
    total_chunks = 50
    chunk_size = math.ceil(len(impact_list)/total_chunks)
    doc_list = []
    for chnk in tqdm(range(total_chunks)):
        curr_set = impact_list[chnk*chunk_size:min((chnk+1)*chunk_size, len(impact_list))]
        text = '\n'.join(curr_set)
        doc = nlp(text)
        doc_list.append(doc)

    # Generating a word cloud with the adjetives of the story
    full_words = ""
    for doc in tqdm(doc_list):
        words = ' '.join(
            [ 
            token.norm_ for token in doc
            if token.is_alpha and not token.like_num and not token.is_stop and
                not token.is_currency and token.pos_ in ['ADJ']
            ]
        )
        full_words = full_words + " " + words

    # stopwords = set(STOPWORDS)
    # text = ' '.join(word for word in text.split() if word not in stopwords)
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title("Identity & Character Perception Impact Word Cloud")
    plt.show()
    return

def get_pwer_dynamics_change_statistics(args):
    content_dict = read_comparisons(args)
    impact_list = []
    for fldr in content_dict:
        for fl, content in content_dict[fldr].items():
            content = content.replace("*", "")
            if "Social Signals & Status" not in content:
                print(content, fldr, fl)
                continue
            impact = content.split("Social Signals & Status")[-1]
            if "4." in impact:
                impact = impact.split("4.")[0]
            if "Perceptual Impact" in impact:
                impact = impact.split("Perceptual Impact")[1]
            impact = impact.split("\n")[0]
            if impact.startswith(":"):
                impact = impact[1:]
            impact_list.append(impact)
    
    # import re
    # from wordcloud import STOPWORDS

    # text = ' '.join(impact_list)

    # text = re.sub(r'[^A-Za-z\s]', '', text)

    # text = text.lower()
    total_chunks = 50
    chunk_size = math.ceil(len(impact_list)/total_chunks)
    doc_list = []
    for chnk in tqdm(range(total_chunks)):
        curr_set = impact_list[chnk*chunk_size:min((chnk+1)*chunk_size, len(impact_list))]
        text = '\n'.join(curr_set)
        doc = nlp(text)
        doc_list.append(doc)

    # Generating a word cloud with the adjetives of the story
    full_words = ""
    for doc in tqdm(doc_list):
        words = ' '.join(
            [ 
            token.norm_ for token in doc
            if token.is_alpha and not token.like_num and not token.is_stop and
                not token.is_currency and token.pos_ in ['ADJ']
            ]
        )
        full_words = full_words + " " + words

    # stopwords = set(STOPWORDS)
    # text = ' '.join(word for word in text.split() if word not in stopwords)
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title("Social Signals & Status Impact Word Cloud")
    plt.show()
    return

def get_narrative_shift_statistics(args):
    content_dict = read_comparisons(args)
    impact_list = []
    for fldr in content_dict:
        for fl, content in content_dict[fldr].items():
            content = content.replace("*", "")
            if "Scene Context & Narrative" not in content:
                print(content, fldr, fl)
                continue
            impact = content.split("Scene Context & Narrative")[-1]
            if "5." in impact:
                impact = impact.split("5.")[0]
            if "Perceptual Impact" in impact:
                impact = impact.split("Perceptual Impact")[1]
            impact = impact.split("\n")[0]
            if impact.startswith(":"):
                impact = impact[1:]
            impact_list.append(impact)
    
    # import re
    # from wordcloud import STOPWORDS

    # text = ' '.join(impact_list)

    # text = re.sub(r'[^A-Za-z\s]', '', text)

    # text = text.lower()
    total_chunks = 50
    chunk_size = math.ceil(len(impact_list)/total_chunks)
    doc_list = []
    for chnk in tqdm(range(total_chunks)):
        curr_set = impact_list[chnk*chunk_size:min((chnk+1)*chunk_size, len(impact_list))]
        text = '\n'.join(curr_set)
        doc = nlp(text)
        doc_list.append(doc)

    # Generating a word cloud with the adjetives of the story
    full_words = ""
    for doc in tqdm(doc_list):
        words = ' '.join(
            [ 
            token.norm_ for token in doc
            if token.is_alpha and not token.like_num and not token.is_stop and
                not token.is_currency and token.pos_ in ['ADJ']
            ]
        )
        full_words = full_words + " " + words

    # stopwords = set(STOPWORDS)
    # text = ' '.join(word for word in text.split() if word not in stopwords)
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title("Scene Context & Narrative Impact Word Cloud")
    plt.show()
    return

def get_manipulation_intent_statistics(args):
    content_dict = read_comparisons(args)
    impact_list = []
    for fldr in content_dict:
        for fl, content in content_dict[fldr].items():
            content = content.replace("*", "")
            if "Manipulation Intent" not in content:
                print(content, fldr, fl)
                continue
            impact = content.split("Manipulation Intent")[-1]
            if "6." in impact:
                impact = impact.split("6.")[0]
            if "Perceptual Impact" in impact:
                impact = impact.split("Perceptual Impact")[1]
            impact = impact.split("\n")[0]
            if impact.startswith(":"):
                impact = impact[1:]
            impact_list.append(impact)
    
    # import re
    # from wordcloud import STOPWORDS

    # text = ' '.join(impact_list)

    # text = re.sub(r'[^A-Za-z\s]', '', text)

    # text = text.lower()
    total_chunks = 50
    chunk_size = math.ceil(len(impact_list)/total_chunks)
    doc_list = []
    for chnk in tqdm(range(total_chunks)):
        curr_set = impact_list[chnk*chunk_size:min((chnk+1)*chunk_size, len(impact_list))]
        text = '\n'.join(curr_set)
        doc = nlp(text)
        doc_list.append(doc)

    # Generating a word cloud with the adjetives of the story
    full_words = ""
    for doc in tqdm(doc_list):
        words = ' '.join(
            [ 
            token.norm_ for token in doc
            if token.is_alpha and not token.like_num and not token.is_stop and
                not token.is_currency and token.pos_ in ['ADJ']
            ]
        )
        full_words = full_words + " " + words

    # stopwords = set(STOPWORDS)
    # text = ' '.join(word for word in text.split() if word not in stopwords)
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title("Manipulation Intent Impact Word Cloud")
    plt.show()
    return

if __name__ == "__main__":
    args = parse_args()
    get_type_statistics(args)
    get_ethical_concern_statistics(args)
    get_emotion_statistics(args)
    get_identity_change_statistics(args)
    get_pwer_dynamics_change_statistics(args)
    get_narrative_shift_statistics(args)
    get_manipulation_intent_statistics(args)

# remove the mild images, analyze only adjectives and adverbs