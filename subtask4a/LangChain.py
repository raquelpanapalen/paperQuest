import os
import json
import pandas as pd
from typing import List, get_type_hints
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests

# Load API key with write access from .env file or environment 
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Check if API key is provided
if not api_key:
    raise ValueError("Please set HUGGINGFACE_API_KEY in the environment or .env file")

# Custom LLM class that works with Hugging Face API
class HuggingFaceChatModel:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def invoke(self, prompt: str):
        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        response = requests.post(url, headers=self.headers, json={"inputs": prompt})
        result = response.json()

        if isinstance(result, list):
            return result[0]["generated_text"]
        elif "generated_text" in result:
            return result["generated_text"]
        elif "error" in result:
            raise ValueError(f"Model Error: {result['error']}")
        else:
            return json.dumps(result)

    def with_structured_output(self, model_class: type(BaseModel)):
        def classify(input_text):
            # Prepare the field description
            def get_model_properties(cls):
                properties = []
                for field_name, field in cls.model_fields.items():
                    description = field.description or field_name
                    properties.append(f"{field_name}: {description}")
                return '\n'.join(properties)

            # Convert model response into Pydantic object
            def store_as_pydantic(content, cls):
                try:
                    content = content.strip().split("\n")[-1]  # Try to extract JSON
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {}
                field_types = get_type_hints(cls)
                parsed_data = {k: parsed.get(k, 0) for k in field_types}
                return cls(**parsed_data)

            # Construct prompt manually
            properties = get_model_properties(model_class)
            prompt = f"""Please classify the tweet below by answering in JSON with 3 binary fields:

{properties}

Tweet:
{input_text}

Return only a valid JSON object with the exact fields and no extra output."""

            # Get model response
            response = self.invoke(prompt)
            return store_as_pydantic(response, model_class)

        return classify

# Classification output schema
class TweetClassification(BaseModel):
    scientific_claim: int = Field(description="1 if the tweet contains a scientific claim, 0 otherwise")
    reference_to_study: int = Field(description="1 if the tweet contains reference to a scientific study/publication, 0 otherwise")
    mentions_scientific_entity: int = Field(description="1 if the tweet mentions a university, scientist, or scientific organization, 0 otherwise")

# Initialize the LLM with Hugging Face API key and model
llm = HuggingFaceChatModel(
    model_name="HuggingFaceH4/zephyr-7b-alpha",  # Supports instruction-following
    api_key=api_key
)

# Build classification function from LLM
classification_fn = llm.with_structured_output(TweetClassification)

# Function to classify tweets
def classify_tweets(tweets: List[str]) -> List[List[int]]:
    results = []
    for i, tweet in enumerate(tweets):
        print(f"Classifying tweet {i + 1}/{len(tweets)}")
        try:
            output = classification_fn(tweet)
            results.append([
                output.scientific_claim,
                output.reference_to_study,
                output.mentions_scientific_entity
            ])
        except Exception as e:
            print(f"Error on tweet {i}: {e}")
            results.append([0, 0, 0])
    return results

# Run classification on ct_test.tsv
if __name__ == "__main__":
    test_df = pd.read_csv("data/ct_test.tsv", sep="\t")
    predictions = classify_tweets(test_df["text"].tolist())

    # Unpack predictions into dataframe
    test_df["cat1_pred"], test_df["cat2_pred"], test_df["cat3_pred"] = zip(*predictions)

    # Save submission
    submission_df = test_df[["index", "cat1_pred", "cat2_pred", "cat3_pred"]]
    submission_df.to_csv("predictions.csv", index=False)
    print("Saved predictions to predictions.csv")
