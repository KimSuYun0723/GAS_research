import pandas as pd

# 1. 원본 TSV 파일 경로
file_path = "C:/_SY/GAS_research/dataset/cola_phenomena/singular_pl.tsv"

# 2. 파일 읽기
# 첫 열에 탭이 추가된 경우 skip_blank_lines=True로 처리
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 3. 불필요한 탭 제거
cleaned_lines = [line.lstrip("\t") for line in lines]

# 4. 클린 데이터 저장
cleaned_file_path = "C:/_SY/GAS_research/dataset/cleaned_data/cleaned_singular_pl.tsv"
with open(cleaned_file_path, "w", encoding="utf-8") as file:
    file.writelines(cleaned_lines)

print(f"클린 데이터가 저장되었습니다: {cleaned_file_path}")