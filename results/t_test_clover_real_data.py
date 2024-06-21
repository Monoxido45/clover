import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import path
import pandas as pd
import scipy.stats as st

# 9 columns in raw data
# first column is normal locart
# second column is random forest locart
# third is difficulty locart
# forth is difficulty random forest locart
# fifth is LCP-RF
# sixth is weighted locart
# seventh is ICP
# eighth is WICP
# and nineth is mondrian split

original_path = os.getcwd()
# plotting object (if needed)
plt.style.use("seaborn-white")
sns.set_palette("tab10")
plt.rcParams.update({"font.size": 12})


# function to construct t-test dataframe
def create_data_t(
    data_name,
    exp_path="/results/pickle_files/real_data_experiments/",
):

    methods = [
        "locart",
        "loforest",
        "A-locart",
        "A-loforest",
        "W-loforest",
        "QRF-TC",
        "reg-split",
        "W-reg-split",
        "mondrian",
    ]

    string_used = "smis"

    # folder path
    folder_path = exp_path + data_name + "_data"
    print(folder_path)

    if path.exists(original_path + folder_path):
        print("Creating data list")
        # list of data frames

        # importing the data
        current_folder = (
            original_path
            + folder_path
            + "/{}_data_score_regression_measures".format(data_name)
        )

        smis_data = np.load(
            current_folder + "/" + string_used + "_{}_data.npy".format(data_name)
        )

        # multiple comparisons with paired-t test
        p_value_list = list()
        method_1, method_2 = list(), list()
        for k in range(0, len(methods)):
            for j in range(k + 1, len(methods)):
                t_array_1, t_array_2 = smis_data[:, k], smis_data[:, j]
                test_array = st.ttest_rel(t_array_1, t_array_2)
                p_value_list.append(test_array[1])
                method_1.append(methods[k])
                method_2.append(methods[j])

        # df to compare mean values (in order to select the best)
        df_smis_values = pd.DataFrame(
            {
                "locart": smis_data[:, 0],
                "loforest": smis_data[:, 1],
                "A-locart": smis_data[:, 2],
                "A-loforest": smis_data[:, 3],
                "QRF-TC": smis_data[:, 4],
                "W-loforest": smis_data[:, 5],
                "reg-split": smis_data[:, 6],
                "W-reg-split": smis_data[:, 7],
                "mondrian": smis_data[:, 8],
            }
        )

        df_smis_values = (
            pd.melt(df_smis_values, var_name="method", value_name="smis_value")
            .groupby("method")
            .mean()
            .reset_index()
        )

        t_df = pd.DataFrame(
            {"p_value": p_value_list, "method_1": method_1, "method_2": method_2}
        )

        t_df["p_value"] = t_df["p_value"].fillna(1)
        t_df.to_csv(original_path + folder_path + "/{}_t_data.csv".format(data_name))

        return t_df, df_smis_values


def create_all_data_t(
    data_list=[
        "winewhite",
        "winered",
        "concrete",
        "airfoil",
        "electric",
        "superconductivity",
        "cycle",
        "protein",
        "news",
        "bike",
        "star",
        "meps19",
        "WEC",
    ],
):
    data_list_t = []
    data_list_smis = []
    for data in data_list:
        # assigning the type of data
        data_t, data_smis = create_data_t(data_name=data)

        data_list_t.append(data_t.assign(data=data))
        data_list_smis.append(data_smis.assign(data=data))
    return data_list_t, data_list_smis


# saving and creating t-data
data_list_t, data_list_smis = create_all_data_t()

# running through data_type
our_method = ["locart", "loforest", "A-locart", "A-loforest", "W-loforest"]
baselines = ["QRF-TC", "mondrian", "reg-split", "W-reg-split"]

# running through data_list to compare our best method with the best baseline
p_value_list = list()
best_method, data_list_names = list(), list()
kind_names = [
    "winewhite",
    "winered",
    "concrete",
    "airfoil",
    "electric",
    "superconductivity",
    "cycle",
    "protein",
    "news",
    "bike",
    "star",
    "meps19",
    "WEC",
]


# overall comparisson
counter = 0
for data_t, data_smis in zip(data_list_t, data_list_smis):

    # checking which of our methods is the best
    our_best_method = (
        data_smis.query("method in @our_method")
        .nlargest(1, "smis_value", keep="first")
        .loc[:, "method"]
        .values
    )

    our_smis = (
        data_smis.query("method in @our_method")
        .nlargest(1, "smis_value", keep="first")
        .loc[:, "smis_value"]
        .values
    )

    # compare to the baselines
    baseline_best_method = (
        data_smis.query("method in @baselines")
        .nlargest(1, "smis_value", keep="first")
        .loc[:, "method"]
        .values
    )

    baseline_smis = (
        data_smis.query("method in @baselines")
        .nlargest(1, "smis_value", keep="first")
        .loc[:, "smis_value"]
        .values
    )

    # returning p-values
    p_value = (
        data_t.query("method_1 in @our_best_method")
        .query("method_2 in @baseline_best_method")["p_value"]
        .values[0]
    )

    p_value_list.append(p_value)
    if our_smis > baseline_smis and p_value < 0.01:
        best_method.append("Our method")
    elif our_smis < baseline_smis and p_value < 0.01:
        best_method.append("Baseline")
    else:
        best_method.append("Tie")
    data_list_names.append(kind_names[counter])
    counter += 1

overall_data_p_values = pd.DataFrame(
    {"p_value": p_value_list, "data": data_list_names, "best": best_method}
)

overall_data_p_values


# obtaining barplot using p-values to count
# comparisson between only conformal methods
counts_dict = {
    "locart": 0,
    "A-locart": 0,
    "reg-split": 0,
    "W-reg-split": 0,
    "mondrian": 0,
}
conformal_methods = ["locart", "A-locart", "reg-split", "W-reg-split", "mondrian"]


for data_t, data_smis in zip(data_list_t, data_list_smis):

    # checking the best method
    best_method = (
        data_smis.query("method in @conformal_methods")
        .nlargest(1, "smis_value", keep="first")
        .loc[:, "method"]
        .values[0]
    )
    counts_dict[best_method] += 1

    for method in conformal_methods:
        if method != best_method:
            best_idx, method_idx = conformal_methods.index(
                best_method
            ), conformal_methods.index(method)
            if best_idx < method_idx:
                data_filtered = data_t.query(
                    "method_1 == @best_method & method_2 == @method"
                )
            else:
                data_filtered = data_t.query(
                    "method_2 == @best_method & method_1 == @method"
                )
            p_val = data_filtered.loc[:, "p_value"].values[0]

            if p_val > 0.01:
                counts_dict[method] += 1

counts_dict

ordered_counts_dict_conf = dict(
    sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)
)


# comparisson between only non-conformal methods
counts_dict = {"loforest": 0, "A-loforest": 0, "W-loforest": 0, "QRF-TC": 0}
non_conf_methods = ["loforest", "A-loforest", "W-loforest", "QRF-TC"]


for data_t, data_smis in zip(data_list_t, data_list_smis):
    # checking the best method
    best_method = (
        data_smis.query("method in @non_conf_methods")
        .nlargest(1, "smis_value", keep="first")
        .loc[:, "method"]
        .values[0]
    )
    counts_dict[best_method] += 1

    for method in non_conf_methods:
        if method != best_method:
            best_idx, method_idx = non_conf_methods.index(
                best_method
            ), non_conf_methods.index(method)
            if best_idx < method_idx:
                data_filtered = data_t.query(
                    "method_1 == @best_method & method_2 == @method"
                )
            else:
                data_filtered = data_t.query(
                    "method_2 == @best_method & method_1 == @method"
                )
            p_val = data_filtered.loc[:, "p_value"].values[0]

            if p_val > 0.01:
                counts_dict[method] += 1

counts_dict
ordered_counts_dict_non_conf = dict(
    sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)
)
ordered_counts_dict_non_conf


# comparisson between all methods
counts_dict = {
    "locart": 0,
    "loforest": 0,
    "A-locart": 0,
    "A-loforest": 0,
    "W-loforest": 0,
    "QRF-TC": 0,
    "reg-split": 0,
    "W-reg-split": 0,
    "mondrian": 0,
}

methods = [
    "locart",
    "loforest",
    "A-locart",
    "A-loforest",
    "W-loforest",
    "QRF-TC",
    "reg-split",
    "W-reg-split",
    "mondrian",
]

for data_t, data_smis in zip(data_list_t, data_list_smis):
    # checking the best method
    best_method = (
        data_smis.nlargest(1, "smis_value", keep="first").loc[:, "method"].values[0]
    )
    counts_dict[best_method] += 1

    for method in methods:
        if method != best_method:
            best_idx, method_idx = methods.index(best_method), methods.index(method)
            if best_idx < method_idx:
                data_filtered = data_t.query(
                    "method_1 == @best_method & method_2 == @method"
                )
            else:
                data_filtered = data_t.query(
                    "method_2 == @best_method & method_1 == @method"
                )
            p_val = data_filtered.loc[:, "p_value"].values[0]

            if p_val > 0.01:
                counts_dict[method] += 1

counts_dict
ordered_counts_dict_all = dict(
    sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)
)
ordered_counts_dict_all


# plotting barplot and saving
images_dir = "results/metric_figures"
results_real = "performance_real/general_results"

# plotting count of data into two barplots
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
ax3.bar(
    x=list(ordered_counts_dict_all.keys()),
    height=list(ordered_counts_dict_all.values()),
    color="tab:blue",
    alpha=0.5,
)
ax3.set_title("All methods")
ax3.set_xlabel("Methods")
ax3.set_ylabel("Frequency")
ax3.tick_params(axis="x", labelrotation=45)
for idx in [0, 1, 3, 4, 7]:
    ax3.get_xticklabels()[idx].set_fontweight("bold")

ax1.bar(
    x=list(ordered_counts_dict_conf.keys()),
    height=list(ordered_counts_dict_conf.values()),
    color="tab:blue",
    alpha=0.5,
)
ax1.set_title("Conformal methods")
ax1.set_xlabel("Methods")
ax1.set_ylabel("Frequency")
ax1.tick_params(axis="x", labelrotation=45)
for idx in [0, 3]:
    ax1.get_xticklabels()[idx].set_fontweight("bold")

ax2.bar(
    x=list(ordered_counts_dict_non_conf.keys()),
    height=list(ordered_counts_dict_non_conf.values()),
    color="tab:blue",
    alpha=0.5,
)
ax2.set_title("Non conformal methods")
ax2.set_xlabel("Methods")
ax2.set_ylabel("Frequency")
ax2.tick_params(axis="x", labelrotation=45)
for idx in [0, 1, 3]:
    ax2.get_xticklabels()[idx].set_fontweight("bold")

plt.tight_layout()
plt.savefig(f"{images_dir}/{results_real}.pdf")


# t-test correlation matrix
mat_dict = {}

methods = [
    "locart",
    "loforest",
    "A-locart",
    "A-loforest",
    "W-loforest",
    "QRF-TC",
    "reg-split",
    "W-reg-split",
    "mondrian",
]

kind_names = [
    "winewhite",
    "winered",
    "concrete",
    "airfoil",
    "electric",
    "superconductivity",
    "cycle",
    "protein",
    "news",
    "bike",
    "star",
    "meps19",
]

counter = 0
for data_t in data_list_t:
    # making diagonal matrix
    cor_mat = np.diag(np.ones(9), k=0)
    for i in range(0, len(methods)):
        for j in range(i + 1, len(methods)):
            method_1, method_2 = methods[i], methods[j]
            data_filtered = data_t.query(
                "method_1 == @method_1 & method_2 == @method_2"
            )
            cor_mat[i, j] = data_filtered.loc[:, "p_value"].values[0]
            cor_mat[j, i] = cor_mat[i, j]
    mat_dict[kind_names[counter]] = cor_mat
    counter += 1

images_dir = "results/metric_figures/performance_real"

first_data_order = ["concrete", "airfoil", "winewhite", "star", "winered", "cycle"]
second_data_order = [
    "electric",
    "bike",
    "meps19",
    "superconductivity",
    "news",
    "protein",
]
list_order = [first_data_order, second_data_order]
# plotting correlation matrix
# creating subplots
# looping through p
count = 0
for i in range(0, 2):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    fig_corr = "heatmaps/p_values_part{}".format(i + 1)
    j = 0
    for data, ax in zip(list_order[i], axs.flatten()):
        cor_dict = mat_dict[data]
        cor_mat = np.round(cor_dict, 3)

        # plotting heatmap
        mask = np.zeros_like(cor_mat)
        mask[np.triu_indices_from(mask, k=1)] = True

        sns.heatmap(
            cor_mat,
            xticklabels=methods,
            yticklabels=methods,
            annot=True,
            cmap="Blues",
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.4},
            annot_kws={"fontsize": 6},
            mask=mask,
        )
        # setting title in each subplot
        ax.title.set_text("data = {}".format(data))
        ax.tick_params(labelsize=8.25)
        count += 1
        j += 1
    # saving figure in correlation folder
    plt.tight_layout()
    plt.savefig(f"{images_dir}/{fig_corr}.pdf")

plt.close()
