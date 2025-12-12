from model_processing.data_functions import load_data, split_dataset, build_vocab, plot_vocab_coverage, analyze_word_frequencies, compute_vocab_coverage

processed_data_path = "./data/processed/sim_sum_gem3_processed.jsonl"
sentences, labels = load_data(processed_data_path)

(train_sentences, train_labels), (val_sentences, val_labels), (test_sentences, test_labels) = split_dataset(sentences, labels)

# Testing different minimum frequency values
min_freq = 5
df = analyze_word_frequencies(sentences=sentences, head_n=5, tail_n=5, min_freq=min_freq)
print(f"With min_freq = {min_freq} then the amount of words removed is {(df['frequency'] < min_freq).sum()} / {len(df)}")

min_freq = 10
df = analyze_word_frequencies(sentences=sentences, head_n=5, tail_n=5, min_freq=min_freq)
print(f"With min_freq = {min_freq} then the amount of words removed is {(df['frequency'] < min_freq).sum()} / {len(df)}")

# Build all vocabularies
word_to_idx_train_all = build_vocab(train_sentences, min_freq=1)
word_to_idx_train_5 = build_vocab(train_sentences, min_freq=5)
word_to_idx_train_10 = build_vocab(train_sentences, min_freq=10)
word_to_idx_val = build_vocab(val_sentences, min_freq=1)

# Compute | Validation ∩ Train | / | Validation |
compute_vocab_coverage(word_to_idx_train_all, word_to_idx_val)
compute_vocab_coverage(word_to_idx_train_5, word_to_idx_val)
compute_vocab_coverage(word_to_idx_train_10, word_to_idx_val)

# Compute | Validation ∩ Train | / | Train |
plot_vocab_coverage(
    denominator_vocabs=[word_to_idx_train_all, word_to_idx_train_5, word_to_idx_train_10],
    numerator_vocabs=[word_to_idx_val, word_to_idx_val, word_to_idx_val],
    titles=["| Validation ∩ Train | / | Train |", "| Validation ∩ Train | / | Train Min_Freq=5 |", "| Validation ∩ Train | / | Train Min_Freq=10 |"],
)
