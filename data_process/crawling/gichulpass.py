import json

from bs4 import BeautifulSoup
import pandas as pd
import requests


def answer_symbol_to_int(symbol: str) -> int:
    answer_map = {"①": 1, "②": 2, "③": 3, "④": 4}
    return answer_map.get(symbol, -1)


def extract_question_data(soup):
    questions = []

    for item in soup.select("#examList > li"):
        # 문제 번호와 질문
        question_text = item.select_one(".pr_problem").get_text(strip=True)

        # 문제 설명 (있는 경우)
        example = item.select_one(".exampleCon")
        example_text = example.get_text(strip=True) if example else ""

        # 선택지를 리스트로 저장
        choices = []
        for choice in item.select(".questionCon li label"):
            choices.append(choice.get_text(strip=True))

        # 정답 번호 추출: "③" 형태로 추출됨
        answer_element = item.select_one(".answer_num")
        answer_text = answer_element.get_text(strip=True).strip() if answer_element else ""
        answer = answer_symbol_to_int(answer_text)

        # 정답 설명
        explanation = item.select_one(".answer_explan")
        explanation_text = explanation.get_text(strip=True) if explanation else ""

        # 데이터 저장
        question_data = {
            "question": question_text,
            "paragraph": example_text,
            "choices": json.dumps(choices, ensure_ascii=False),
            "answer": answer,
            "answer_explanation": explanation_text,
        }
        questions.append(question_data)

    return pd.DataFrame(questions)


def crawl_and_save(subject_code):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    url = "https://gichulpass.com/bbs/board.php"

    # 경제학 문제
    if subject_code == 27:
        wr_ids = []

    # 행정법 문제
    if subject_code == 32:
        wr_ids = []

    # 행정학 문제
    if subject_code == 33:
        wr_ids = []

    # 한국사 문제
    if subject_code == 34:
        wr_ids = []

    # 사회 문제
    if subject_code == 35:
        wr_ids = []
        wr_ids += list(range(808, 814)) + [908]
        wr_ids += list(range(814, 819)) + [1010, 865]
        wr_ids += list(range(819, 824)) + [1050, 839]
        wr_ids += list(range(824, 833)) + [894]
        wr_ids += list(range(833, 836)) + [1027]
        wr_ids += list(range(836, 839))

    dfs = []
    for wr_id in wr_ids:
        params = {"bo_table": "exam", "wr_id": wr_id, "subject": subject_code}
        response = requests.get(url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        df = extract_question_data(soup)
        dfs.append(df)

    concated_df = pd.concat(dfs, axis=0)
    len_df = len(concated_df)
    concated_df.to_csv(f"gichulpass_{subject_code}_{len_df}.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    crawl_and_save(subject_code=35)