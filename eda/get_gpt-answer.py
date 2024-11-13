from openai import OpenAI
import pandas as pd
import re
import ast
import time
from loguru import logger

# 로그 파일 설정
logger.add("gpt_answers_log.log", rotation="1 MB")  # 1MB 넘을 시 새 파일 생성

# OpenAI API 클라이언트 초기화
client = OpenAI(api_key="")

data = "../data/train.csv"
model = "gpt-4o"

# CSV 파일 읽기
df = pd.read_csv(data)

# 새로운 컬럼 생성
df[f"{model}_answer"] = None

def generate_answer(paragraph, question, choices):
    # GPT-3.5 Turbo 모델 프롬프트 구성
    choices_text = "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
    prompt = f"""
    다음은 객관식 문제입니다. 제시된 지문과 질문을 읽고 정답을 선택하십시오.

    지문: {paragraph}

    질문: {question}
    선택지:
    {choices_text}

    정답을 선택지 번호로만 답하시오.
    단하나 정답의 번호만 반드시 도출하시오.
    정답 번호 이외의 다른 숫자는 출력하지 마시오.
    """

    # OpenAI API를 사용하여 답변 생성
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0,  # 온도 설정
        max_tokens=8,     # 응답 토큰 길이 제한
    )

    # 모델의 응답을 번호로 추출
    answer = response.choices[0].message.content.strip()
    request_id = response.id
    return answer, request_id  # request ID 반환

# 각 행에 대해 API 호출 및 답변 저장
for index, row in df.iterrows():
    paragraph = row["paragraph"]
    problem_data = ast.literal_eval(row["problems"])
    question = problem_data["question"]
    choices = problem_data["choices"]

    try:
        answer, request_id = generate_answer(paragraph, question, choices)
        answer_parsed = re.search(r'\b[1-5]\b', answer)  # 1에서 4까지의 숫자만 찾음

        if answer_parsed:
            answer_parsed = answer_parsed.group()  # 매칭된 첫 번째 숫자 추출
        else:
            answer_parsed = None  # 매칭이 없는 경우 None으로 설정
            logger.warning(f"Index {index}: No valid answer found in response - {answer}")

        logger.info(f"Index {index}: Answer generated - {answer}, parsed answer - {answer_parsed}, request_id - {request_id}")
        df.at[index, f"{model}_answer"] = answer_parsed
    except Exception as e:
        logger.error(f"Error at index {index}: {e}")

    # API 호출 제한 준수를 위해 대기 시간 추가
    time.sleep(1)

# CSV 파일로 저장
output_path = data.replace(".csv", f"_add-{model}-answer.csv")
df.to_csv(output_path, index=False)
logger.info(f"Results saved to {output_path}")
