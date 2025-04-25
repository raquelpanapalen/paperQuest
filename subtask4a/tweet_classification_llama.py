import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field
import os 
from sklearn.metrics import f1_score, classification_report
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM

INPUT_PATH = "data/ct_dev.tsv"
OUTPUT_DIR = "output/"
OUTPUT_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")

class TweetClassification(BaseModel):
    scientific_claim: Literal[0, 1] = Field(..., description="Does the tweet contain a scientific claim?")
    scientific_reference: Literal[0, 1] = Field(..., description="Does the tweet reference a scientific study or publication?")
    scientific_entity: Literal[0, 1] = Field(..., description="Does the tweet mention a scientific entity like a university or scientist?")


parser = PydanticOutputParser(pydantic_object=TweetClassification)

# Escape curly braces
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


model = OllamaLLM(model="llama3")

chain = prompt | model | parser


input_df = pd.read_csv("data/ct_dev.tsv", sep="\t")
predictions = []

for _, row in tqdm(input_df.iterrows(), total=len(input_df)):
    tweet_text = row["text"]
    try:
        result = chain.invoke({"tweet_text": tweet_text})
        predictions.append({
            "index": row["index"],
            "cat1_pred": result.scientific_claim,
            "cat2_pred": result.scientific_reference,
            "cat3_pred": result.scientific_entity,
        })
    except Exception as e:
        print(f"Failed on index {row['index']}: {e}")
        predictions.append({
            "index": row["index"],
            "cat1_pred": 0,
            "cat2_pred": 0,
            "cat3_pred": 0,
        })

predictions_df = pd.DataFrame(predictions)

# Save predictions
predictions_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PREDICTIONS_PATH}")

# Load ground truth
ground_truth_df = pd.read_csv(INPUT_PATH, sep="\t")

# Merge predictions with ground truth on 'index'
merged_df = pd.merge(ground_truth_df, predictions_df, on="index")

# Extract true and predicted labels
y_true = merged_df[["cat1", "cat2", "cat3"]].values
y_pred = merged_df[["cat1_pred", "cat2_pred", "cat3_pred"]].values

# Macro-averaged F1 Score
macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"\nMacro-Averaged F1 Score: {macro_f1:.4f}")

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("output/predictions.csv", index=False)

print("Predictions saved to predictions.csv")
