# main.py

import os
import argparse
import torch
import torch.distributed as dist

from data_loaders import DataLoader
from inference import InferenceModel
from loguru import logger
from model import ModelHandler
from trainer import CustomTrainer
from utils import (
    GoogleDriveManager,
    create_experiment_filename,
    load_config,
    load_env_file,
    log_config,
    set_logger,
    set_seed,
)
import wandb


def main():
    # 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="분산 학습을 위한 로컬 랭크"
    )
    args = parser.parse_args()

    # 분산 환경 초기화
    if args.local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 환경 설정 및 시드 설정
    load_env_file()
    config = load_config(args.config)
    set_logger(log_file=config["log"]["file"], log_level=config["log"]["level"])
    set_seed()

    # wandb 설정 (메인 프로세스에서만 초기화)
    exp_name = create_experiment_filename(config)
    if args.local_rank == 0 or args.local_rank == -1:
        wandb.init(
            config=config,
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            name=exp_name,
        )

    # wandb 실험명으로 config 갱신
    config["training"]["run_name"] = exp_name
    config["inference"]["output_path"] = os.path.join(
        config["inference"]["output_path"], exp_name + "_output.csv"
    )
    log_config(config)

    try:
        # 모델 및 토크나이저 설정
        model_handler = ModelHandler(config["model"])
        model, tokenizer = model_handler.setup()
        model.to(device)

        # 학습용 데이터 처리
        data_processor = DataLoader(tokenizer, config["data"])
        train_dataset, eval_dataset = data_processor.prepare_datasets(is_train=True)
        test_dataset = data_processor.prepare_datasets(is_train=False)

        # 학습
        trainer = CustomTrainer(
            training_config=config["training"],
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trained_model = trainer.train()

        # 추론
        inferencer = InferenceModel(
            inference_config=config["inference"],
            model=trained_model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
        )
        inferencer.run_inference()

    except Exception as e:
        logger.exception(f"오류 발생: {e}")
        if args.local_rank == 0 or args.local_rank == -1:
            wandb.finish(exit_code=1)
    else:
        if args.local_rank == 0 or args.local_rank == -1:
            logger.info("출력 및 설정 파일을 Google Drive에 업로드합니다...")
            gdrive_manager = GoogleDriveManager()
            gdrive_manager.upload_exp(
                config["exp"]["username"],
                config["inference"]["output_path"],
            )
            wandb.finish()


if __name__ == "__main__":
    main()
