import pandas as pd
import time
from data_processing.prompt import system_prompt
from data_processing.functions import extract_metadata, filter_text, llm, validate_bio_format, parse_bio_string, save_row_to_jsonl

test_samples = 20
raw_data_path = "./data/raw/SimSUM.csv"

models = {
    "llama": {
        "output_path": "./data/processed/experiment/llama_processed.jsonl",
        "model_name": "llama3:8b",
        "completion_time": None,
        "success_count": 0,
        "failure_count": 0
    },
    "gemma": {
        "output_path": "./data/processed/experiment/gemma_processed.jsonl",
        "model_name": "gemma3:12b",
        "completion_time": None,
        "success_count": 0,
        "failure_count": 0
    }, 
    "qwen": {
        "output_path": "./data/processed/experiment/qwen_processed.jsonl",
        "model_name": "qwen3:8b",
        "completion_time": None,
        "success_count": 0,
        "failure_count": 0
    }
}

df_simsum = pd.read_csv(raw_data_path, sep=';', engine='python', index_col=0)
features = df_simsum.columns.to_list()[:-2]

for key, info in models.items():

    print(f"Processing with model: {key}")
    start = time.time()

    for index, row in df_simsum.iterrows():

        if index >= test_samples:
            break
        
        print("Row: ", index)
        metadata, text = extract_metadata(row, features), filter_text(row["text"])
        input_prompt = "Text: " + text + " Metadata: " + str(metadata)
        llm_output = llm(system_prompt, input_prompt, ollama_model=info["model_name"])
        
        if validate_bio_format(llm_output):
            print(f"[SUCCESS] - Row {index} passed BIO format validation.")
            info["success_count"] += 1

            tokens, labels = parse_bio_string(llm_output)

            save_row_to_jsonl(info['output_path'], index, tokens, labels)
            print(f"[SAVED] - Row {index} saved to {info['output_path']}.\n")
        else:
            info["failure_count"] += 1

    end = time.time()
    info["completion_time"] = end - start
    print(f"Processing with model {key} completed in {end - start:.2f} seconds.\n")

for key, info in models.items():
    print(f"Model: {key}, Completion Time: {info['completion_time']:.2f} seconds")
    print(f"Model: {key}, Success Count: {info['success_count']}, Failure Count: {info['failure_count']}\n")