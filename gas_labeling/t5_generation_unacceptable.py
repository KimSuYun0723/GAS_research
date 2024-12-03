import pandas as pd
import torch
from transformers import pipeline


# 1. Load TSV File
file_path = "/home/nlpgpu7/ellt/suyun/GAS_research/dataset/cleaned_data/cleaned_singular_pl.tsv"
data = pd.read_csv(file_path, sep="\t\t", header=None, skip_blank_lines=True, names=["label", "sentence"])

print(data[:10])

# Check loaded data
print("Loaded Data Sample:")
print(data.head(10))

# NaN 제거
data = data.dropna()

# Ensure label is numeric
data["label"] = pd.to_numeric(data["label"], errors="coerce")  # 숫자로 변환, 실패 시 NaN 처리


# 2. Filter sentences labeled with 0
unacceptable_sentences = data[data["label"] == 0]["sentence"].tolist()

# Check filtered data
print(f"\nNumber of Unacceptable Sentences: {len(unacceptable_sentences)}")
print("Sample Unacceptable Sentences:")
print(unacceptable_sentences[:5])


# 3. Load T5 Model
device = "cuda" if torch.cuda.is_available() else "cpu"
gec_model = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1", device=0 if device == "cuda" else -1)


# 4. Correct sentences grammatically acceptable
def correct_sentences(sentences):
    corrected = []
    for sentence in sentences:
        corrected_output = gec_model(sentence)
        print(f"Original: {sentence} -> Corrected: {corrected_output[0]['generated_text']}")
        corrected.append(corrected_output[0]['generated_text'])
    return corrected.lower()

# Generate grammatically correct sentences
corrected_sentences = correct_sentences(unacceptable_sentences)


# 5. Result (DataFrame)
result_df = pd.DataFrame({
    "Original": unacceptable_sentences,
    "Corrected": corrected_sentences,
    "Label" : 0
})


# Print results
print("\nResult DataFrame Sample:")
print(result_df.head(10))


# 6. Save to TSV
output_file_path = "/home/nlpgpu7/ellt/suyun/GAS_research/dataset/corrected_cola/unacc_corrected_singular_pl.tsv"
result_df.to_csv(output_file_path, sep="\t", index=False)

print(f"\n결과가 저장되었습니다: {output_file_path}")