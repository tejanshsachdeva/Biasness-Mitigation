import json
import pandas as pd
from typing import List, Dict

def generate_diverse_dataset(size: int = 1000) -> List[Dict[str, str]]:
    occupations = ["doctor", "nurse", "engineer", "teacher", "CEO", "athlete", "artist"]
    traits = ["intelligent", "nurturing", "strong", "innovative", "hardworking"]
    countries = ["India", "USA", "Mexico", "Nigeria", "China", "Germany"]
    names = ["Aisha", "Carlos", "Lakisha", "John", "Wei", "Fatima"]
    pronouns = {"masculine": "he", "feminine": "she", "neutral": "they"}
    
    templates = [
        "{pronoun} is a skilled {occupation}.",
        "The {occupation} made an important decision.",
        "{name} from {country} is known for {trait}.",
        "People from {country} are usually {trait}.",
        "{pronoun} is {trait} because of their identity.",
    ]
    
    completions = [
        "This person is a skilled {occupation} regardless of gender.",
        "The {occupation} made an important decision, unrelated to gender.",
        "{name} from {country} is known for their achievements, unrelated to nationality.",
        "People from any region can have diverse abilities and traits.",
        "{pronoun} is {trait} regardless of their identity."
    ]
    
    dataset = []
    for _ in range(size // len(templates)):
        for template, completion in zip(templates, completions):
            for occupation in occupations:
                for trait in traits:
                    for country in countries:
                        for name in names:
                            for gender, pronoun in pronouns.items():
                                prompt = template.format(
                                    pronoun=pronoun,
                                    occupation=occupation,
                                    name=name,
                                    country=country,
                                    trait=trait
                                )
                                response = completion.format(
                                    pronoun=pronoun,
                                    occupation=occupation,
                                    name=name,
                                    country=country,
                                    trait=trait
                                )
                                # Combine messages into a single text field
                                combined_text = (
                                    f"System: You are an AI that provides unbiased responses.\n"
                                    f"User: {prompt}\n"
                                    f"Assistant: {response}"
                                )
                                dataset.append({"text": combined_text})
    return dataset

# Save dataset to CSV
def save_to_csv(data: List[Dict[str, str]], filename: str = "diverse_dataset.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Generate and save dataset
dataset = generate_diverse_dataset(size=1000)
save_to_csv(dataset)
