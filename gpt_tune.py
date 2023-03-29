import csv
import openai
import pandas as pd
import subprocess
import os


def prepare_data():
    completions = pd.read_csv("data/completions.csv")
    semantics = pd.read_csv("data/semantic_sims.csv")

    # combine the two data frames vertically
    combined_df = pd.concat([completions, semantics], axis=1)

    # sort the resulting data frame based on the values in the 'semantics' column
    sorted_df = combined_df.sort_values('semantic_sim')
    sorted_df = sorted_df[sorted_df['semantic_sim'] > 0.3]
    sorted_df.loc[:, ["prompt", "completion"]].to_csv(
        "data/prepared_data.csv", index=False)


def fine_tune():
    openai.api_key = "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"
    subprocess.run(
        'openai tools fine_tunes.prepare_data --file Data/prepared_data.csv --quiet'.split())
    # Start fine-tuning
    subprocess.run(
        'openai api fine_tunes.create --training_file Data/prepared_data_prepared.jsonl --model davinci --suffix "IVM_generator"'.split())


def main():
    prepare_data()
    fine_tune()


if __name__ == "__main__":
    main()
