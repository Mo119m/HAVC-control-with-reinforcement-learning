import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_and_plot_rewards(
    paths,
    first_n=10,
    y_min=None,
    y_max=None,
    output_png="rewards_plot.png",
    return_dataframe=False,
):

    records = []
    for p in paths:
        name = os.path.basename(p)
        with open(p, "r") as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, dict) and "reward" in item:
                records.append({
                    "file": name,
                    "step": item.get("step", None),
                    "reward": item["reward"],
                })

    df = pd.DataFrame(records)

    topn = df.groupby("file").apply(lambda g: g.sort_values("step").head(first_n)).reset_index(drop=True)

    plt.figure()
    for file_name, sub in topn.groupby("file"):
        sub_sorted = sub.sort_values("step")
        plt.plot(sub_sorted["step"], sub_sorted["reward"], marker="o", label=file_name)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"First {first_n} Rewards per File")
    if (y_min is not None) and (y_max is not None):
        plt.ylim(y_min, y_max)
    plt.legend()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()

    if return_dataframe:
        return output_png, topn
    return output_png


# png_path, df = extract_and_plot_rewards(
#     paths=["mini_rollout_fewshot_7B_finetuned.json",
#            "mini_rollout_fewshot.json",
#            "mini_rollout_llm_good.json"],
#     first_n=15,
#     y_min=-100,
#     y_max=0,
#     output_png="rewards_plot.png",
#     return_dataframe=True
# )
