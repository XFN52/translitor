import requests
import json
from sacrebleu.metrics import BLEU, CHRF, TER
import sys
import time

def evaluate_translation(url, source_text, reference_translation, source_lang="ru", target_lang="en"):
    """
    Оценивает качество перевода с использованием метрик BLEU, chrF и TER
    """
    try:
        response = requests.post(
            f"{url}/translate",
            json={
                "text": source_text,
                "source_lang": source_lang,
                "target_lang": target_lang
            },
            timeout=30  # Увеличенный таймаут для работы через ngrok
        )
        response.raise_for_status()
        machine_translation = response.json()["translation"]
        
        # Инициализация метрик
        bleu = BLEU()
        chrf = CHRF()
        ter = TER()
        
        # Расчет метрик
        bleu_score = bleu.corpus_score([machine_translation], [[reference_translation]])
        chrf_score = chrf.corpus_score([machine_translation], [[reference_translation]])
        ter_score = ter.corpus_score([machine_translation], [[reference_translation]])
        
        return {
            "machine_translation": machine_translation,
            "bleu": float(bleu_score.score),
            "chrf": float(chrf_score.score),
            "ter": float(ter_score.score)
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Ошибка при запросе к API: {str(e)}"}

def main():
    if len(sys.argv) < 2:
        print("Использование: python evaluate_translations.py <ngrok_url>")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    # Тестовые случаи
    test_cases = [
        {
            "name": "Короткое предложение",
            "source": "Привет, как дела?",
            "reference": "Hi, how are you?",
            "source_lang": "ru",
            "target_lang": "en"
        },
        {
            "name": "Длинный текст",
            "source": "Искусственный интеллект - это способность компьютерных систем выполнять задачи, которые обычно требуют человеческого интеллекта. Это включает в себя распознавание речи, принятие решений, визуальное восприятие и перевод между языками.",
            "reference": "Artificial intelligence is the ability of computer systems to perform tasks that typically require human intelligence. This includes speech recognition, decision-making, visual perception, and translation between languages.",
            "source_lang": "ru",
            "target_lang": "en"
        },
        {
            "name": "Технические термины",
            "source": "Python - высокоуровневый язык программирования с динамической типизацией и автоматическим управлением памятью.",
            "reference": "Python is a high-level programming language with dynamic typing and automatic memory management.",
            "source_lang": "ru",
            "target_lang": "en"
        },
        {
            "name": "Числа и даты",
            "source": "В 2023 году компания инвестировала 1.5 миллиона долларов в развитие технологий.",
            "reference": "In 2023, the company invested 1.5 million dollars in technology development.",
            "source_lang": "ru",
            "target_lang": "en"
        },
        {
            "name": "Имена собственные",
            "source": "Илон Маск основал компании Tesla и SpaceX.",
            "reference": "Elon Musk founded Tesla and SpaceX.",
            "source_lang": "ru",
            "target_lang": "en"
        },
        {
            "name": "Перевод с английского на русский",
            "source": "The quick brown fox jumps over the lazy dog.",
            "reference": "Быстрая коричневая лиса прыгает через ленивую собаку.",
            "source_lang": "en",
            "target_lang": "ru"
        },
        {
            "name": "Вопрос (EN-RU)",
            "source": "What is the weather like today?",
            "reference": "Какая сегодня погода?",
            "source_lang": "en",
            "target_lang": "ru"
        },
        {
            "name": "Длинное предложение с техническими терминами (EN-RU)",
            "source": "Machine learning algorithms enable computers to learn from data and improve their performance over time without being explicitly programmed.",
            "reference": "Алгоритмы машинного обучения позволяют компьютерам учиться на данных и со временем улучшать свою производительность без явного программирования.",
            "source_lang": "en",
            "target_lang": "ru"
        },
        {
            "name": "Числа и единицы (EN-RU)",
            "source": "The temperature is 25 degrees Celsius, and the humidity is 60%.",
            "reference": "Температура составляет 25 градусов Цельсия, а влажность - 60%.",
            "source_lang": "en",
            "target_lang": "ru"
        },
        {
            "name": "Идиома (EN-RU)",
            "source": "It's raining cats and dogs.",
            "reference": "Льет как из ведра.",
            "source_lang": "en",
            "target_lang": "ru"
        }
    ]
    
    print(f"Тестирование переводчика на {base_url}\n")
    
    total_bleu = 0
    total_chrf = 0
    total_ter = 0
    successful_tests = 0
    
    for test_case in test_cases:
        print(f"Тест: {test_case['name']}")
        print(f"Исходный текст: {test_case['source']}")
        print(f"Эталонный перевод: {test_case['reference']}")
        
        result = evaluate_translation(
            base_url,
            test_case['source'],
            test_case['reference'],
            test_case['source_lang'],
            test_case['target_lang']
        )
        
        if "error" in result:
            print(f"Ошибка: {result['error']}\n")
            continue
            
        print(f"Машинный перевод: {result['machine_translation']}")
        print(f"BLEU: {result['bleu']:.2f}")
        print(f"chrF: {result['chrf']:.2f}")
        print(f"TER: {result['ter']:.2f}\n")
        
        total_bleu += result['bleu']
        total_chrf += result['chrf']
        total_ter += result['ter']
        successful_tests += 1
        
        # Небольшая пауза между запросами
        time.sleep(1)
    
    if successful_tests > 0:
        print("Средние показатели:")
        print(f"BLEU: {total_bleu/successful_tests:.2f}")
        print(f"chrF: {total_chrf/successful_tests:.2f}")
        print(f"TER: {total_ter/successful_tests:.2f}")
    else:
        print("Не удалось выполнить ни одного теста успешно.")

if __name__ == "__main__":
    main() 