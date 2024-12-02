# Levenshtein Distance
import pandas as pd
from Levenshtein import distance as levenshtein_distance

input_file = "/home/nlpgpu7/ellt/suyun/GAS_research/dataset/corrected_cola/unacc_corrected_singular_pl.tsv"
output_file = "/home/nlpgpu7/ellt/suyun/GAS_research/dataset/corrected_cola/unacc_with_distance.tsv"

# Normalized Edit Distance 계산 함수
def normalized_edit_distance(original, corrected):
    edit_distance = levenshtein_distance(original, corrected)
    max_length = max(len(original), len(corrected))
    return edit_distance / max_length

# 데이터 로드
df = pd.read_csv(input_file, sep="\t")

# Edit Distance 계산 및 새로운 열 추가
df['Edit_Distance'] = df.apply(lambda row: normalized_edit_distance(row['Original'], row['Corrected']), axis=1)

# 결과 저장
df.to_csv(output_file, sep="\t", index=False)

print(f"Saved with Edit Distance: {output_file}")