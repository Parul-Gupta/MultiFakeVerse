import os
import pandas as pd
import argparse
import glob
import csv
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("analyze the mask generation and image-level comparison outputs")
    parser.add_argument("--mask_metrics_folder", "-m", type=str, default="outputs/mask_metrics")
    args = parser.parse_args()
    return args

def get_aggregate_statistics(args):
    file_list = glob.glob(f"{args.mask_metrics_folder}/**/*.jsonl", recursive=True)
    df_list = []
    for fl in file_list:
        df = pd.read_json(fl, lines=True)
        df_list.append(df)
    full_df = pd.concat(df_list, axis=0)
    full_df = full_df.fillna(0)
    avg_columns = ["post_edit_ratio", "ssim", "mse", "lpips_score", "largest_component_size", "cc_clusters", "cluster_dist"]

    formal_names = ["Edit Area Ratio", "SSIM score", "PSNR", "LPIPS_Score", "Largest Component Size", "# Connected Clusters", "Cluster distance"]

    for idx, col in enumerate(avg_columns):
        full_df[col] = full_df[col].replace('NA', 0)
        lst = full_df[col].to_list()
        # lst = [i for i in lst if i!='NA']
        if col == "post_edit_ratio":
            # print(min(lst), max(lst))
            # lst = [i*100 for i in lst]
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")
            sns.histplot(data=full_df, x="post_edit_ratio", bins=10, stat="percent")
            plt.xlabel("Ratio of Edited Area", fontsize=20)
            plt.ylabel("Percentage of Images", fontsize=20)
            plt.tight_layout()
            # plt.show()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.savefig("plots/edited_area_norm_distribution.pdf")

            # with open("post_edit.csv", 'w', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(lst)
        elif col == "mse":
            lst1 = []
            for i in lst:
                if i!=0:
                    lst1.append(20 * np.log10(255.0 / (np.sqrt(i))))
                else:
                    lst1.append(0)
            
            lst = lst1
        print(f"Average {formal_names[idx]}: {sum(lst)/len(lst)}")
    
    return

if __name__ == "__main__":
    args = parse_args()
    get_aggregate_statistics(args)