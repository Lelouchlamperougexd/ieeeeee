# import openai
import pandas as pd
from apis.weather_api import get_weather_data
from apis.traffic_api import get_traffic_data
from models.model_utils import load_t5_model, load_bart_model
from models.inference import inference_with_time
from metrics.bleu_rouge import calculate_bleu, calculate_rouge

# API GPT-3
# openai.api_key = 'your-openai-api-key'

# def generate_response_openai(prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=100,
#         temperature=0.7
#     )
#     return response['choices'][0]['text'].strip()

def run_experiment(location="Astana"):
    """Run the full experiment with weather and traffic data."""
    # Fetch context data
    weather_context = get_weather_data(location)
    traffic_context = get_traffic_data(location)

    # Generate enriched queries
    query = f"How can I get from A to B? Context: {traffic_context}; {weather_context}."
    reference = "The best route is through XYZ, avoiding heavy traffic."

    # Load models
    model_t5, tokenizer_t5 = load_t5_model()
    model_bart, tokenizer_bart = load_bart_model()

    # Generate responses
    response_t5, time_t5 = inference_with_time(model_t5, tokenizer_t5, query)
    response_bart, time_bart = inference_with_time(model_bart, tokenizer_bart, query)
    # response_openai = generate_response_openai(query)

    # Evaluate results
    bleu_t5 = calculate_bleu(reference, response_t5)
    bleu_bart = calculate_bleu(reference, response_bart)
    # bleu_openai = calculate_bleu(reference, response_openai)

    rouge_t5 = calculate_rouge(reference, response_t5)
    rouge_bart = calculate_rouge(reference, response_bart)
    # rouge_openai = calculate_rouge(reference, response_openai)

    # Print results
    print(f"T5 Response: {response_t5} | Time: {time_t5:.4f}s | BLEU: {bleu_t5:.4f} | ROUGE: {rouge_t5}")
    print(f"BART Response: {response_bart} | Time: {time_bart:.4f}s | BLEU: {bleu_bart:.4f} | ROUGE: {rouge_bart}")
    # print(f"OpenAI GPT-3 Response: {response_openai} | BLEU: {bleu_openai:.4f} | ROUGE: {rouge_openai}")

if __name__ == "__main__":
    run_experiment(location="Astana")