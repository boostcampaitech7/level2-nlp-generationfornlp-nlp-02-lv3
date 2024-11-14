import argparse
import os

from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .util import load_env_file


class HuggingFaceHubManager:
    def __init__(self):
        load_env_file("../config/.env")
        self.token = os.getenv("HF_TOKEN")
        self.organization = os.getenv("HF_TEAM_NAME")
        self.project_name = os.getenv("HF_PROJECT_NAME")

    def upload_model(self, model_name, username, checkpoint_path):
        repo_id = f"{model_name}-{username}"
        try:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

            model.push_to_hub(repo_id=repo_id, organization=self.organization, use_auth_token=self.token)
            tokenizer.push_to_hub(repo_id=repo_id, organization=self.organization, use_auth_token=self.token)
            logger.debug(f"your model pushed successfully in {self.repo_id}, hugging face")
        except Exception as e:
            logger.debug(f"An error occurred while uploading to Hugging Face: {e}")

    def upload_dataset(self, file_name, private=True):
        """
        폴더 내 데이터 파일들을 Hugging Face Hub에 데이터셋으로 업로드하는 함수.

        Parameters:
        - file_name (str): 업로드할 로컬 데이터 파일 이름
        - token (str): Hugging Face 액세스 토큰. 쓰기 권한 필요
        - private (bool): True면 비공개, False면 공개 설정

        Returns:
        - None
        """
        api = HfApi()
        repo_id = f"{self.organization}/{self.project_name}-{file_name}"

        # 리포지토리 존재 여부 확인
        try:
            api.repo_info(repo_id, repo_type="dataset", token=self.token)
            logger.debug(f"'{repo_id}' 리포지토리가 이미 존재합니다. 기존 리포지토리에 데이터셋을 업로드합니다.")
        except Exception:
            # 리포지토리가 없으면 생성
            logger.debug(f"'{repo_id}' 리포지토리가 존재하지 않습니다. 새로 생성한 후 업로드합니다.")
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, token=self.token)

        # 파일 경로 설정
        file_path = os.path.join("..", "data", f"{file_name}.csv")
        if not os.path.exists(file_path):
            logger.debug(f"파일 '{file_path}'이 존재하지 않습니다.")
            return

        # 데이터셋 로드 및 업로드
        dataset = load_dataset("csv", data_files={"train": file_path})
        dataset.push_to_hub(repo_id, token=self.token)
        logger.debug(f"데이터셋이 '{repo_id}'에 업로드되었습니다.")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="./outputs/your_path")
    parser.add_argument("--modelname", type=str, default="your-modelname")
    parser.add_argument("--dataname", type=str, default="your-dataname")
    parser.add_argument("--username", type=str, default="your-username")

    hf_manager = HuggingFaceHubManager()
    hf_manager.upload_model(parser.modelname, parser.username, parser.checkpoint_path)
    # hf_manager.upload_dataset(parser.dataname, private=True)