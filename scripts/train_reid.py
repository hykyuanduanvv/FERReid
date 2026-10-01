import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import transformers
from adapters.args_reid import ReIDTrainingArguments as TrainingArguments


def main():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Seed model initialization as well as the later Trainer sampling.
    transformers.set_seed(args.seed)

    if args.no_cuda:
        device = "cpu"
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from adapters.trainer_reid import DGReIDTrainer
    trainer = DGReIDTrainer(args=args, device=device)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()  # trainer_state.json (full log history) in output_dir


if __name__ == "__main__":
    main()
