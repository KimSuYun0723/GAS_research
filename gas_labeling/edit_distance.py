# 편집거리 말고... 형태소 분석해서 자른다음에 
# 비교해야할 것 같은데...? 
# 삽입 삭제 대체? 
import spacy
from Levenshtein import distance as levenshtein_distance

# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

def pos_based_edit_distance(original, corrected):
    # 형태소 및 품사 추출
    original_tokens = [(token.text, token.pos_) for token in nlp(original)]
    corrected_tokens = [(token.text, token.pos_) for token in nlp(corrected)]

    # 품사 포함 토큰 출력 (디버깅용)
    print(f"Original tokens with POS: {original_tokens}")
    print(f"Corrected tokens with POS: {corrected_tokens}")

    # 형태소와 품사를 조합해 문자열로 변환
    original_combined = ["_".join(token) for token in original_tokens]
    corrected_combined = ["_".join(token) for token in corrected_tokens]

    # Edit Distance 계산
    return levenshtein_distance(" ".join(original_combined), " ".join(corrected_combined))

# 예제 실행
original_sentence = "the boy are here"
corrected_sentence = "the boy is here"

distance = pos_based_edit_distance(original_sentence, corrected_sentence)
print(f"POS-based Edit Distance: {distance}")

