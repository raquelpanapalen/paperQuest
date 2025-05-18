import pandas as pd
from tqdm import tqdm
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM


class TweetClassification(BaseModel):
    scientific_claim: Literal[0, 1] = Field(description="1 if the tweet contains a scientific claim, 0 otherwise")
    scientific_reference: Literal[0, 1] = Field(description="1 if the tweet contains a scientific reference or publication, 0 otherwise")
    scientific_entity: Literal[0, 1] = Field(description="1 if the tweet mentions a university, scientist, or scientific organization, 0 otherwise")

def run_llama_inference(input_path="data/ct_dev.tsv", output_path="output/llama_predictions.csv", model_name="llama3"):
    model = OllamaLLM(model=model_name)
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

    chain = prompt | model | parser

    df = pd.read_csv(input_path, sep="\t")
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running LLaMA inference"):
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

    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print(f"LLaMA predictions saved to {output_path}")
