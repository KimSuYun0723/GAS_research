import pandas as pd
import torch
from transformers import pipeline


# 1. Load TSV File
file_path = "/home/nlpgpu7/ellt/suyun/GAS_research/dataset/cleaned_data/cleaned_singular_pl.tsv"
data = pd.read_csv(file_path, sep="\t\t", header=None, skip_blank_lines=True, names=["label", "sentence"])

# NaN 제거
data = data.dropna()

# Ensure label is numeric
data["label"] = pd.to_numeric(data["label"], errors="coerce")  # 숫자로 변환, 실패 시 NaN 처리

# 2. Filter sentences labeled with 0
acceptable_sentences = data[data["label"] == 1]["sentence"].tolist()

# Check filtered data
print(f"\nNumber of Unacceptable Sentences: {len(acceptable_sentences)}")
print("Sample Unacceptable Sentences:")
print(acceptable_sentences[:5])


# 3. Load T5 Model
device = "cuda" if torch.cuda.is_available() else "cpu"
gec_model = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1", device=0 if device == "cuda" else -1)

# 4. Correct sentences grammatically acceptable
def correct_sentences(sentences):
    corrected = []
    for sentence in sentences:
        corrected_output = gec_model(sentence, max_new_tokens=40)
        print(f"Original: {sentence} -> Corrected: {corrected_output[0]['generated_text']}")
        corrected.append(corrected_output[0]['generated_text'])
    return corrected

# Generate grammatically correct sentences
corrected_sentences = correct_sentences(acceptable_sentences)


# 5. Result (DataFrame)
result_df = pd.DataFrame({
    "Original": acceptable_sentences,
    "Corrected": corrected_sentences,
    "Label" : 1
})


# Print results
print("\nResult DataFrame Sample:")
print(result_df.head(10))


# 6. Save to TSV
output_file_path = "/home/nlpgpu7/ellt/suyun/GAS_research/dataset/corrected_cola/acc_corrected_singular_pl.tsv"
result_df.to_csv(output_file_path, sep="\t", index=False)

print(f"\n결과가 저장되었습니다: {output_file_path}")