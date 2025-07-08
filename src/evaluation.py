import os
import time
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from PIL import Image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_text(text):
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_bleu(reference, prediction):
    ref_tokens = reference.strip().split()
    pred_tokens = prediction.strip().split()
    if len(pred_tokens) == 0:
        return 0.0
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)


def compute_rouge(reference, prediction):
    if not prediction.strip() or not reference.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def predict(image_path, question, model, vocab, device, max_len=20, transform=None):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    image = image.unsqueeze(0).to(device)

    question_encoded = vocab.encode(normalize_text(question))
    question_tensor = torch.tensor(question_encoded).unsqueeze(0).to(device)

    sos_token = vocab.word2idx["<SOS>"]
    eos_token = vocab.word2idx["<EOS>"]
    answer_input = torch.tensor([[sos_token]]).to(device)
    generated = []

    with torch.no_grad():
        for _ in range(max_len):
            output = model(image, question_tensor, answer_input)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            token_id = next_token.item()
            if token_id == eos_token:
                break
            generated.append(token_id)
            answer_input = torch.cat([answer_input, next_token.unsqueeze(0)], dim=1)

    return vocab.decode(generated)


def predict_from_tensor(image_tensor, question, model, vocab, device, max_len=20):
    model.eval()
    image = image_tensor.unsqueeze(0).to(device)
    question_encoded = vocab.encode(normalize_text(question))
    question_tensor = torch.tensor(question_encoded).unsqueeze(0).to(device)

    sos_token = vocab.word2idx["<SOS>"]
    eos_token = vocab.word2idx["<EOS>"]
    answer_input = torch.tensor([[sos_token]]).to(device)
    generated = []

    with torch.no_grad():
        for _ in range(max_len):
            output = model(image, question_tensor, answer_input)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            token_id = next_token.item()
            if token_id == eos_token:
                break
            generated.append(token_id)
            answer_input = torch.cat([answer_input, next_token.unsqueeze(0)], dim=1)

    return vocab.decode(generated)


def evaluate_model(model, model_name, test_dataset, vocab, device, top_k=10):
    results = []
    model.eval()
    random_samples = random.sample(test_dataset.samples, top_k)

    print(f"\nĐánh giá mô hình {model_name} trên {top_k} mẫu ngẫu nhiên:\n")
    with torch.no_grad():
        for i, (img_path, question, ref_answer) in enumerate(random_samples):
            pred_answer = predict(img_path, question, model, vocab, device, transform=test_dataset.transform)
            ref_text = normalize_text(ref_answer)
            bleu = compute_bleu(ref_text, pred_answer)
            rouge = compute_rouge(ref_text, pred_answer)

            results.append({
                "Image": os.path.basename(img_path),
                "Question": question,
                "Ref": ref_text,
                "Pred": pred_answer,
                "BLEU": round(bleu, 4),
                "ROUGE-L": round(rouge, 4),
            })

            print(f"[{i+1}] {question}")
            print(f"Predict: {pred_answer}")
            print(f"Ref: {ref_text}")
            print(f"BLEU: {bleu:.4f} | ROUGE-L: {rouge:.4f}\n")

    df = pd.DataFrame(results)
    print(df[["Ảnh", "BLEU", "ROUGE-L"]])

    avg_bleu = df["BLEU"].mean()
    avg_rouge = df["ROUGE-L"].mean()
    print(f"\nBLEU Avg: {avg_bleu:.4f} | ROUGE-L Avg: {avg_rouge:.4f}")

    return df


def evaluate_on_testset(model, test_loader, vocab, device, samples=200):
    model.eval()
    avg_bleu, avg_rouge, count = 0, 0, 0

    with torch.no_grad():
        for images, questions, answers in test_loader:
            for i in range(min(10, images.size(0))):
                image = images[i].to(device)
                question_text = vocab.decode(questions[i].cpu().numpy())
                true_answer_text = vocab.decode(answers[i].cpu().numpy())

                pred_text = predict_from_tensor(image, question_text, model, vocab, device)

                bleu = compute_bleu(true_answer_text, pred_text)
                rouge = compute_rouge(true_answer_text, pred_text)

                avg_bleu += bleu
                avg_rouge += rouge
                count += 1

                if count >= samples:
                    break
            if count >= samples:
                break

    if count > 0:
        avg_bleu /= count
        avg_rouge /= count
        print(f"BLEU Score: {avg_bleu:.4f} | ROUGE-L Score: {avg_rouge:.4f}")
    else:
        print("No valid sentence to evaluate")

    return avg_bleu, avg_rouge


def compare_models(model_results):
    model_results = sorted(model_results, key=lambda x: x["bleu"], reverse=True)
    table_data = []
    for result in model_results:
        table_data.append({
            "Model": result["name"],
            "Params": result["params"],
            "Train time (s)": result["train_time"],
            "BLEU Score": result["bleu"],
            "ROUGE-L Score": result["rouge"]
        })

    df = pd.DataFrame(table_data)
    print(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_results))
    width = 0.35
    ax.bar(x - width/2, [r["bleu"] for r in model_results], width, label='BLEU')
    ax.bar(x + width/2, [r["rouge"] for r in model_results], width, label='ROUGE-L')
    ax.set_ylabel('Score')
    ax.set_title('BLEU and ROUGE-L')
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in model_results])
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, [r["train_time"] for r in model_results], width)
    ax.set_ylabel('Time (sec)')
    ax.set_title('Time training')
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in model_results])
    plt.tight_layout()
    plt.show()

    return df
