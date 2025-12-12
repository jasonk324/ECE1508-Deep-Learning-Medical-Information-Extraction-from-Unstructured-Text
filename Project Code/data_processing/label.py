import pandas as pd
import json
from data_processing.prompt import system_prompt
from data_processing.functions import extract_metadata, filter_text, llm, validate_bio_format, parse_bio_string, save_row_to_jsonl

raw_data_path = "./data/raw/SimSUM.csv"
output_path = "./data/processed/sim_sum_gem3_processed.jsonl"

df_simsum = pd.read_csv(raw_data_path, sep=';', engine='python', index_col=0)
features = df_simsum.columns.to_list()[:-2]

processed_data = [json.loads(line) for line in open(output_path)]
if len(processed_data) > 0:
    start_index = processed_data[-1]['index'] + 1
else:
    start_index = 0

for index, row in df_simsum.iterrows():

    if index < start_index:
        continue
    
    print("Row: ", index)
    metadata, text = extract_metadata(row, features), filter_text(row["text"])
    input_prompt = "Text: " + text + " Metadata: " + str(metadata)
    llm_output = llm(system_prompt, input_prompt)
    
    if validate_bio_format(llm_output):
        print(f"[SUCCESS] - Row {index} passed BIO format validation.")

        tokens, labels = parse_bio_string(llm_output)

        save_row_to_jsonl(output_path, index, tokens, labels)
        print(f"[SAVED] - Row {index} saved to {output_path}.\n")