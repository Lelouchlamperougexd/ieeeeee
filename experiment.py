# import openai
import pandas as pd
from models.model_utils import load_t5_model, load_bart_model
from models.inference import inference_with_time
from metrics.bleu_rouge import calculate_bleu, calculate_rouge

# Установите API ключ для OpenAI GPT-3
# openai.api_key = 'your-openai-api-key'

# def generate_response_openai(prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=100,
#         temperature=0.7
#     )
#     return response['choices'][0]['text'].strip()

def run_experiment():
    # Загрузка моделей
    model_t5, tokenizer_t5 = load_t5_model()
    model_bart, tokenizer_bart = load_bart_model()

    # Тестовые данные (например, запрос)
    query = "How can I get from point A to point B faster right now?"
    reference = "The best route is through XYZ, considering traffic."


    # Получение ответа для каждой модели
    response_t5, time_t5 = inference_with_time(model_t5, tokenizer_t5, query)
    response_bart, time_bart = inference_with_time(model_bart, tokenizer_bart, query)
    # response_openai = generate_response_openai(query)

    # Оценка качества
    bleu_score_t5 = calculate_bleu(reference, response_t5)
    bleu_score_bart = calculate_bleu(reference, response_bart)
    # bleu_score_openai = calculate_bleu(reference, response_openai)

    rouge_score_t5 = calculate_rouge(reference, response_t5)
    rouge_score_bart = calculate_rouge(reference, response_bart)
    # rouge_score_openai = calculate_rouge(reference, response_openai)

    # Результаты
    print(f"Response from T5: {response_t5} | Time: {time_t5:.4f}s | BLEU: {bleu_score_t5:.4f} | ROUGE: {rouge_score_t5}")
    print(f"Response from BART: {response_bart} | Time: {time_bart:.4f}s | BLEU: {bleu_score_bart:.4f} | ROUGE: {rouge_score_bart}")
    # print(f"Response from OpenAI: {response_openai} | Time: N/A | BLEU: {bleu_score_openai:.4f} | ROUGE: {rouge_score_openai}")


if __name__ == "__main__":
    run_experiment()
