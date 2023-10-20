from scipy import stats
import os

def read_metrics(file_path):
    metrics = {'map': [], 'ndcg_cut_10': [], 'recall_1000': []}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] in metrics:
                if parts[1]=="all":
                    continue
                metrics[parts[0]].append(float(parts[2]))
    return metrics



def perform_ttest(data1, data2):
    t_stat, p_value = stats.ttest_rel(data1, data2)
    return p_value < CORRECTED_SIGNIFICANCE


def read_all_metrics():
    all_results = {}
    for dataset in datasets:
        dataset_results = {}
        for final_name in final_names:
            file_path = f'evaluation/{dataset}/{final_name}/metrics.txt'
            if os.path.exists(file_path):
                dataset_results[final_name] = read_metrics(file_path)
        all_results[dataset] = dataset_results
    return all_results


if __name__ == '__main__':
    datasets = ["trec_covid", "robust04"]
    final_names = ["Baseline&&none","&&Krovetz","&&Porter","VS&&ChatGPT","&&LlaMa-2","CS&&LlaMa-2","ECS.1&Roberta&Porter","1&Roberta&ChatGPT","1&Roberta&LlaMa-2","1&LlaMa-2&Porter","1&LlaMa-2&ChatGPT","1&LlaMa-2&LlaMa-2","ECS.2&Roberta&Porter","2&Roberta&ChatGPT","2&Roberta&LlaMa-2","2&LlaMa-2&Porter","2&LlaMa-2&ChatGPT","2&LlaMa-2&LlaMa-2"]

    all_results = read_all_metrics()
    CORRECTED_SIGNIFICANCE = 0.05 / len(final_names)
    for final_name in final_names:
        row = f"{final_name}"
        for dataset in datasets:
            dataset_results = all_results.get(dataset, {})
            metrics = dataset_results.get(final_name, {})

            for metric_name in ['map', 'ndcg_cut_10', 'recall_1000']:
                value = metrics.get(metric_name, ["N/A"])
                avg_value = sum(value) / len(value) if value != ["N/A"] else "N/A"

                if avg_value != "N/A":
                    significant = ""
                    if perform_ttest(value, all_results[dataset]["&&Porter"].get(metric_name, [0])):
                        significant = "*"
                    row += f" & {avg_value:.4f}{significant}"
                else:
                    row += f" & {avg_value}"
        row += " \\\\"
        print(row)


