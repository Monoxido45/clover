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
    kind,
    p=np.array([1, 3, 5]),
    exp_path="/results/pickle_files/locart_all_metrics_experiments/",
    other_asym=False,
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

    string_used = "mean_diff"

    # folder path
    folder_path = exp_path + kind + "_data"
    print(folder_path)
    if other_asym:
        folder_path = folder_path + "_eta_1.5"

    if path.exists(original_path + folder_path):
        print("Creating data list")
        # list of data frames
        t_list = list()
        mean_values_list = list()
        for i in range(p.shape[0]):
            # importing the data
            current_folder = (
                original_path
                + folder_path
                + "/{}_score_regression_p_{}_10000_samples_measures".format(kind, p[i])
            )

            mean_dif_data = np.load(
                current_folder
                + "/"
                + string_used
                + "_p_{}_{}_data.npy".format(p[i], kind)
            )

            # multiple comparisons with paired-t test
            p_value_list = list()
            method_1, method_2 = list(), list()
            for k in range(0, len(methods)):
                for j in range(k + 1, len(methods)):
                    t_array_1, t_array_2 = mean_dif_data[:, k], mean_dif_data[:, j]
                    test_array = st.ttest_rel(t_array_1, t_array_2)
                    p_value_list.append(test_array[1])
                    method_1.append(methods[k])
                    method_2.append(methods[j])

            # df to compare mean values (in order to select the best)
            df_mean_values = pd.DataFrame(
                {
                    "locart": mean_dif_data[:, 0],
                    "loforest": mean_dif_data[:, 1],
                    "A-locart": mean_dif_data[:, 2],
                    "A-loforest": mean_dif_data[:, 3],
                    "QRF-TC": mean_dif_data[:, 4],
                    "W-loforest": mean_dif_data[:, 5],
                    "reg-split": mean_dif_data[:, 6],
                    "W-reg-split": mean_dif_data[:, 7],
                    "mondrian": mean_dif_data[:, 8],
                }
            )

            df_mean_values = (
                pd.melt(df_mean_values, var_name="method", value_name="mean_diff_value")
                .groupby("method")
                .mean()
                .reset_index()
                .assign(p=p[i])
            )

            t_df = pd.DataFrame(
                {"p_value": p_value_list, "method_1": method_1, "method_2": method_2}
            ).assign(p=p[i])

            t_df["p_value"] = t_df["p_value"].fillna(1)

            t_list.append(t_df)
            mean_values_list.append(df_mean_values)

        # doing the same to correlation data
        data_t_final = pd.concat(t_list)
        mean_values_final = pd.concat(mean_values_list)
        data_t_final.to_csv(original_path + folder_path + "/{}_t_data.csv".format(kind))

        return data_t_final, mean_values_final


def create_all_data_t(
    kind_list=[
        "homoscedastic",
        "heteroscedastic",
        "asymmetric",
        "asymmetric_V2",
        "t_residuals",
        "non_cor_heteroscedastic",
        "splitted_exp",
        "correlated_heteroscedastic",
    ],
    p=np.array([1, 3, 5]),
):
    data_list_t = []
    data_list_diff = []
    for kind in kind_list:
        if kind == "asymmetric_V2":
            other_asym = True
            kind = "asymmetric"
        else:
            other_asym = False

        # assigning the type of data
        data_t, data_diff = create_data_t(kind, p=p, other_asym=other_asym)

        data_list_t.append(data_t.assign(data_type=kind))
        data_list_diff.append(data_diff.assign(data_type=kind))
    return data_list_t, data_list_diff


# saving and creating t-data
data_list_t, mean_values_list = create_all_data_t(p=np.array([1, 3, 5]))

# running through data_type
p_chosen = [1, 3, 5]
our_method = ["locart", "loforest", "A-locart", "A-loforest", "W-loforest"]
baselines = ["QRF-TC", "mondrian", "reg-split", "W-reg-split"]

# running through data_list to compare our best method with the best baseline
p_value_list, p_list = list(), list()
best_method, kind_list = list(), list()
kind_names = [
    "homoscedastic",
    "heteroscedastic",
    "asymmetric",
    "asymmetric_V2",
    "t_residuals",
    "non_cor_heteroscedastic",
    "splitted_exp",
    "correlated_heteroscedastic",
]


# overall comparisson
counter = 0
for data_t, data_diff in zip(data_list_t, mean_values_list):
    for p in p_chosen:
        data_t_p, data_diff_p = data_t.query("p == @p"), data_diff.query("p == @p")

        # checking which of our methods is the best
        our_best_method = (
            data_diff_p.query("method in @our_method")
            .nsmallest(1, "mean_diff_value", keep="first")
            .loc[:, "method"]
            .values
        )

        our_diff = (
            data_diff_p.query("method in @our_method")
            .nsmallest(1, "mean_diff_value", keep="first")
            .loc[:, "mean_diff_value"]
            .values
        )

        # compare to the baselines
        baseline_best_method = (
            data_diff_p.query("method in @baselines")
            .nsmallest(1, "mean_diff_value", keep="first")
            .loc[:, "method"]
            .values
        )

        baseline_diff = (
            data_diff_p.query("method in @baselines")
            .nsmallest(1, "mean_diff_value", keep="first")
            .loc[:, "mean_diff_value"]
            .values
        )

        # returning p-values
        p_value = (
            data_t_p.query("method_1 in @our_best_method")
            .query("method_2 in @baseline_best_method")["p_value"]
            .values[0]
        )
        p_value_list.append(p_value)
        if our_diff < baseline_diff and p_value < 0.01:
            best_method.append("Our method")
        elif our_diff > baseline_diff and p_value < 0.01:
            best_method.append("Baseline")
        else:
            best_method.append("Tie")
        kind_list.append(kind_names[counter])
        p_list.append(p)
    counter += 1

overall_data_p_values = pd.DataFrame(
    {"p_value": p_value_list, "kind": kind_list, "p": p_list, "best": best_method}
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


for data_t, data_diff in zip(data_list_t, mean_values_list):
    for p in p_chosen:
        data_t_p, data_diff_p = data_t.query("p == @p"), data_diff.query("p == @p")

        # checking the best method
        best_method = (
            data_diff_p.query("method in @conformal_methods")
            .nsmallest(1, "mean_diff_value", keep="first")
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
                    data_filtered = data_t_p.query(
                        "method_1 == @best_method & method_2 == @method"
                    )
                else:
                    data_filtered = data_t_p.query(
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


for data_t, data_diff in zip(data_list_t, mean_values_list):
    for p in p_chosen:
        data_t_p, data_diff_p = data_t.query("p == @p"), data_diff.query("p == @p")
        # checking the best method
        best_method = (
            data_diff_p.query("method in @non_conf_methods")
            .nsmallest(1, "mean_diff_value", keep="first")
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
                    data_filtered = data_t_p.query(
                        "method_1 == @best_method & method_2 == @method"
                    )
                else:
                    data_filtered = data_t_p.query(
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

for data_t, data_diff in zip(data_list_t, mean_values_list):
    for p in p_chosen:
        data_t_p, data_diff_p = data_t.query("p == @p"), data_diff.query("p == @p")
        # checking the best method
        best_method = (
            data_diff_p.nsmallest(1, "mean_diff_value", keep="first")
            .loc[:, "method"]
            .values[0]
        )
        counts_dict[best_method] += 1

        for method in methods:
            if method != best_method:
                best_idx, method_idx = methods.index(best_method), methods.index(method)
                if best_idx < method_idx:
                    data_filtered = data_t_p.query(
                        "method_1 == @best_method & method_2 == @method"
                    )
                else:
                    data_filtered = data_t_p.query(
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
results_p = "performance_vs_p/general_results"

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
for idx in [0, 1, 2, 3, 5]:
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
for idx in [0, 1]:
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
for idx in [0, 1, 2]:
    ax2.get_xticklabels()[idx].set_fontweight("bold")

plt.tight_layout()
plt.savefig(f"{images_dir}/{results_p}.pdf")


# t-test correlation matrix
cor_mat_dict_list = []

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

for data_t in data_list_t:
    mat_dict = {}
    for p in p_chosen:
        data_t_p = data_t.query("p == @p")
        # making diagonal matrix
        cor_mat = np.diag(np.ones(9), k=0)
        for i in range(0, len(methods)):
            for j in range(i + 1, len(methods)):
                method_1, method_2 = methods[i], methods[j]
                data_filtered = data_t_p.query(
                    "method_1 == @method_1 & method_2 == @method_2"
                )
                cor_mat[i, j] = data_filtered.loc[:, "p_value"].values[0]
                cor_mat[j, i] = cor_mat[i, j]
        mat_dict[p] = cor_mat
    cor_mat_dict_list.append(mat_dict)


# plotting correlation matrix
# creating subplots
# looping through p
count = 0
for kind in kind_names:
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 8))
    cor_dict = cor_mat_dict_list[count]
    fig_corr = "heatmaps/p_values_{}".format(kind)
    for p_sel, ax in zip(p_chosen, axs):
        cor_mat = np.round(cor_dict[p_sel], 3)

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
        ax.title.set_text("p = {}".format(p_sel))
        ax.tick_params(labelsize=8.25)
    count += 1
    # saving figure in correlation folder
    plt.tight_layout()
    plt.savefig(f"{images_dir}/{fig_corr}.pdf")

plt.close()

# plotting heatmap
sns.heatmap(
    cor_mat,
    xticklabels=cor_mat.columns,
    yticklabels=cor_mat.columns,
    annot=True,
    cmap="Blues",
    ax=ax,
    square=True,
    cbar_kws={"shrink": 0.4},
)
# setting title in each subplot
ax.title.set_text("p = {}".format(p_sel))
ax.tick_params(labelsize=8.25)


# saving figure in correlation folder
plt.tight_layout()
plt.savefig(f"{images_dir}/{fig_corr}.pdf")
