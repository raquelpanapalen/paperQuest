import argparse
from debert import run_bert_training
from llama import run_llama_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["debert", "llama"], required=True, help="Choose which model to run.")
    args = parser.parse_args()

    if args.model == "bert":
        run_bert_training()
    elif args.model == "llama":
        run_llama_inference()
