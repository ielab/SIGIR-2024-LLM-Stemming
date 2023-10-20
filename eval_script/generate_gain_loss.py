from scipy import stats
import os
import matplotlib.pyplot as plt


name_dict = {
"&&Porter": "Porter",
"1&LlaMa-2&Porter": "ECS.1_LlaMa-2_Porter",
"2&LlaMa-2&Porter": "ECS.2_LlaMa-2_Porter"
}
# Step 1: Extract the required data
def read_metrics(file_path):
    metrics = {'map': {}, 'ndcg_cut_10': {}, 'recall_1000': {}}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[1]=="all":
                continue
            if parts[0] in metrics:
                metrics[parts[0]][parts[1]] = float(parts[2])  # store using qid as key
    return metrics


def read_all_metrics():
    datasets = ["trec_covid", "robust04"]
    final_names = ["Baseline&&none","&&Krovetz","&&Porter","VS&&ChatGPT","&&LlaMa-2","CS&&LlaMa-2","ECS.1&Roberta&Porter","1&Roberta&ChatGPT","1&Roberta&LlaMa-2","1&LlaMa-2&Porter","1&LlaMa-2&ChatGPT","1&LlaMa-2&LlaMa-2","ECS.2&Roberta&Porter","2&Roberta&ChatGPT","2&Roberta&LlaMa-2","2&LlaMa-2&Porter","2&LlaMa-2&ChatGPT","2&LlaMa-2&LlaMa-2"]

    all_results = {}
    for dataset in datasets:
        dataset_results = {}
        for final_name in final_names:
            file_path = f'evaluation/{dataset}/{final_name}/metrics.txt'
            if os.path.exists(file_path):
                dataset_results[final_name] = read_metrics(file_path)
        all_results[dataset] = dataset_results
    return all_results


# Step 2: Generate and save the gain-loss plot
def generate_gain_loss_plot(result_1, result_2, all_results, dataset, metric_name):
    result1_metrics = all_results[dataset].get(result_1, {}).get(metric_name, {})
    result2_metrics = all_results[dataset].get(result_2, {}).get(metric_name, {})

    # Sort the qids numerically
    qids = sorted(result1_metrics.keys(), key=int)

    differences = [(result1_metrics.get(qid, 0) - result2_metrics.get(qid, 0)) for qid in qids]

    plt.figure(figsize=(50, 6))
    plt.bar(range(len(differences)), differences, color=['g' if d > 0 else 'r' for d in differences])
    #plt.axhline(0, color="black", linestyle="--")
    plt.xticks(range(len(differences)), labels=qids, rotation=90)
    #plt.xticks(range(len(differences)), labels=[""] * len(differences), rotation=90)
    #plt.title(f"Gain-Loss Plot: {name_dict[result_1]} vs. {name_dict[result_2]} on {dataset}")
    plt.ylabel(f"{metric_name.replace('_cut_', '@').upper()} Difference", fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=14)
    plt.xlabel(f"Queries", fontsize=24)

    plt.ylim(-0.4, 0.4)
    plt.tight_layout()

    # Create directory structure if it doesn't exist
    directory = f"graph_{name_dict[result_1]}_{name_dict[result_2]}/{dataset}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f"{directory}/{metric_name}_gain_loss_plot.pdf")


if __name__ == '__main__':
    all_results = read_all_metrics()
    result_2 = "&&Porter"
    result_1 = "1&LlaMa-2&Porter"
    # result_1 = "2&LlaMa-2&Porter"
    # result_2 = "1&LlaMa-2&Porter"
    # result_2 = "&&Porter"
    # result_1 = "2&LlaMa-2&Porter"
    for dataset in ["trec_covid", "robust04"]:
        for metric in ['map', 'ndcg_cut_10', 'recall_1000']:
            generate_gain_loss_plot(result_1, result_2, all_results, dataset, metric)

