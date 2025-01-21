import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Инференс для заданной модели
def generate_response(model, tokenizer, query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True)  # Явно задаем max_length
    output = model.generate(inputs['input_ids'], max_length=100)  # Ограничиваем длину выхода
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Замер времени инференса
def inference_with_time(model, tokenizer, query):
    start_time = time.time()
    response = generate_response(model, tokenizer, query)
    end_time = time.time()
    time_taken = end_time - start_time
    return response, time_taken
