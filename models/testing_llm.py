from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer (you can swap for a different model if needed)
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Or another instruct model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create pipeline for text generation
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)


# Function to extract keywords
def extract_keywords(text):
    prompt = (
        f"Extract concise and relevant keywords from the following text for improved document retrieval:\n\n"
        f'"{text}"\n\n'
        f"Keywords:"
    )

    output = llm(prompt, max_new_tokens=60, do_sample=False)[0]["generated_text"]

    # Extract keywords from response
    keywords_start = output.find("Keywords:") + len("Keywords:")
    keywords = output[keywords_start:].strip().split(",")
    return [kw.strip() for kw in keywords if kw.strip()]


# Example usage
text = """
This study isn't receiving sufficient attention. It reveals Black/Latino/Indigenous individuals aren't just succumbing to COVID at higher rates than Whites but are also passing away earlier. 
90% of White fatalities occur in those 65+, 90% of Black fatalities occur in those 55+, and 89% of Indigenous fatalities occur in those 45+.
"""

keywords = extract_keywords(text)
print("Extracted Keywords:", keywords)
