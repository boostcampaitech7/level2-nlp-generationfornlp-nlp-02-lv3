import ast

import pandas as pd


def calculate_accuracy(model):
    # CSV 파일 읽기
    df = pd.read_csv(f"../data/train_add-{model}-answer.csv")

    # problems 컬럼에서 original_answer 추출
    df["original_answer"] = df["problems"].apply(lambda x: ast.literal_eval(x)["answer"])

    # 모델 예측과 원래 정답 비교
    df["is_correct"] = df["original_answer"].astype(str) == df[f"{model}_answer"].astype(str)

    # 정확도 계산
    correct_count = df["is_correct"].sum()
    total_count = len(df)
    accuracy = correct_count / total_count * 100

    # 결과 출력
    print(f"Model: {model}")
    print(f"Total Questions: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    calculate_accuracy("gpt-3.5-turbo")
    calculate_accuracy("gpt-4o")
    calculate_accuracy("gpt-4o-mini")
