import os
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM
from sklearn.metrics import f1_score
import time
import json
from llamaapi import LlamaAPI

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5 

# Define the TweetClassification model
class TweetClassification(BaseModel):
    scientific_claim: Literal[0, 1] = Field(..., description="Does the tweet contain a scientific claim?")
    scientific_reference: Literal[0, 1] = Field(..., description="Does the tweet reference a scientific study or publication?")
    scientific_entity: Literal[0, 1] = Field(..., description="Does the tweet mention a scientific entity like a university or scientist?")

def run_llama_inference(input_path: str, output_dir: str = "output", model_name: str = "llama3") -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.csv")

    # Initialize the LlamaAPI with your API token
    llama = LlamaAPI(API)

    # Set up the LLaMA model and prompt
    parser = PydanticOutputParser(pydantic_object=TweetClassification)
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_template(f"""
    You are an expert in analyzing social media posts.

    Classify the following tweet by answering:
    1. Does it contain a scientific claim?
    2. Does it reference a scientific study or publication?
    3. Does it mention a scientific entity (e.g., a university or scientist)?

    Return your answer in this JSON format:
    {format_instructions}

    Tweet: {{tweet_text}}
    """)

    model = OllamaLLM(model=model_name)
    chain = prompt | model | parser

    df = pd.read_csv(input_path, sep="\t")
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running LLaMA inference"):
        tweet_text = row["text"]
        attempts = 0
        success = False
        while attempts < MAX_RETRIES and not success:
            try:
                # Execute LLaMA inference
                api_request_json = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": tweet_text},
                    ],
                    "stream": False,
                }

                response = llama.run(api_request_json)
                result = response.json()  # Get the response as JSON

                # Assuming result is a list, get the first item (adjust if necessary based on your actual response)
                if isinstance(result, list):
                    result = result[0]  # Adjust this based on the actual structure you see

                # Access attributes safely using .get to avoid KeyError
                predictions.append({
                    "index": row["index"],
                    "cat1_pred": result.get("scientific_claim", 0),
                    "cat2_pred": result.get("scientific_reference", 0),
                    "cat3_pred": result.get("scientific_entity", 0),
                })
                success = True
            except Exception as e:
                attempts += 1
                print(f"Failed on index {row['index']}: {e}")
                if attempts < MAX_RETRIES:
                    print(f"Retrying... (Attempt {attempts}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Max retries reached for index {row['index']}. Skipping this entry.")
                    predictions.append({
                        "index": row["index"],
                        "cat1_pred": 0,
                        "cat2_pred": 0,
                        "cat3_pred": 0,
                    })

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Evaluate if ground truth is available
    if "cat1" in df.columns:
        merged_df = pd.merge(df, predictions_df, on="index")
        y_true = merged_df[["cat1", "cat2", "cat3"]].values
        y_pred = merged_df[["cat1_pred", "cat2_pred", "cat3_pred"]].values
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Macro-Averaged F1 Score: {macro_f1:.4f}")
