#!/bin/bash


trec_eval=$1
# Define arrays for datasets, run_files, and final_names
datasets=("trec_covid" "robust04")
run_files=("test-default.tsv" "test-krovetz.tsv" "test-porter.tsv" "openai_vocab_stemmed.tsv" "llama_vocab_stemmed.tsv" "llama_context.tsv" "roberta_entities_porter_replace.tsv" "roberta_entities_openai_replace.tsv" "roberta_entities_llama_replace.tsv" "llama_entities_porter_replace.tsv"  "llama_entities_openai_replace.tsv" "llama_entities_llama_replace.tsv" "roberta_entities_porter_append.tsv" "roberta_entities_openai_append.tsv" "roberta_entities_llama_append.tsv" "llama_entities_porter_append.tsv"  "llama_entities_openai_append.tsv" "llama_entities_llama_append.tsv")
final_names=("Baseline&&none" "&&Krovetz" "&&Porter" "VS&&ChatGPT" "&&LlaMa-2" "CS&&LlaMa-2" "ECS.1&Roberta&Porter" "1&Roberta&ChatGPT" "1&Roberta&LlaMa-2" "1&LlaMa-2&Porter" "1&LlaMa-2&ChatGPT" "1&LlaMa-2&LlaMa-2" "ECS.2&Roberta&Porter" "2&Roberta&ChatGPT" "2&Roberta&LlaMa-2" "2&LlaMa-2&Porter" "2&LlaMa-2&ChatGPT" "2&LlaMa-2&LlaMa-2")

# Metrics to evaluate
metrics=("ndcg_cut.10" "map" "recall.1000")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Create a folder for each dataset under evaluation
    mkdir -p "evaluation/$dataset"

    # Define the path to the Qrels file for the current dataset
    qrels_path="topics/$dataset/qrels.txt"

    # Loop through each run file
    for index in "${!run_files[@]}"; do
        run_file="${run_files[$index]}"
        final_name="${final_names[$index]}"

        # Create a folder for each final name under the dataset folder
        mkdir -p "evaluation/$dataset/$final_name"

        # Define the path to the run file for the current dataset and run_file
        run_file_path="runs/$dataset/$run_file"

        # Define the path to the output file for the current dataset and final_name
        output_file_path="evaluation/$dataset/$final_name/metrics.txt"

        # Run trec_eval and save the output to the evaluation folder
        $trec_eval -q -m ${metrics[0]} -m ${metrics[1]} -m ${metrics[2]} $qrels_path $run_file_path > $output_file_path

        # Print out a message indicating that the evaluation is done for this configuration
        echo "Evaluated $dataset with $final_name, results saved to $output_file_path"
    done
done
