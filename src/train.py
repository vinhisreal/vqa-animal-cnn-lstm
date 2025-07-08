# src/train.py
import time
import torch
import os
import torch.nn as nn
import torch.optim as optim
from evaluation import predict_from_tensor, compute_bleu, compute_rouge, count_parameters

def train_model(model, model_name, vocab, train_loader, test_loader, device, num_epochs=15):
    os.makedirs("weights", exist_ok=True)  # tạo thư mục nếu chưa có
    print(f"\n========== {model_name} ==========")
    print(f"Number of trainable parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    history = {"train_loss": [], "val_bleu": [], "val_rouge": []}
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, questions, answers in train_loader:
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)

            answer_input = answers[:, :-1]
            answer_target = answers[:, 1:].contiguous()

            optimizer.zero_grad()
            outputs = model(images, questions, answer_input)

            loss = criterion(outputs.view(-1, outputs.size(-1)), answer_target.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Validation
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            total_bleu = 0
            total_rouge = 0
            count = 0

            with torch.no_grad():
                for images, questions, answers in test_loader:
                    for i in range(min(5, len(images))):
                        img = images[i].to(device)
                        q_text = vocab.decode(questions[i].cpu().numpy())
                        a_text = vocab.decode(answers[i].cpu().numpy())

                        pred = predict_from_tensor(img, q_text, model, vocab, device)
                        total_bleu += compute_bleu(a_text, pred)
                        total_rouge += compute_rouge(a_text, pred)
                        count += 1
                    if count >= 100:
                        break

            bleu_score = total_bleu / count
            rouge_score = total_rouge / count
            history["val_bleu"].append(bleu_score)
            history["val_rouge"].append(rouge_score)
            print(f"Epoch {epoch+1:02d}: Loss={avg_loss:.4f} | BLEU={bleu_score:.4f} | ROUGE-L={rouge_score:.4f}")
        else:
            print(f"Epoch {epoch+1:02d}: Loss={avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f'weights/{model_name}_epoch_{epoch+1}.pth')

    training_time = time.time() - start_time
    print(f"\n>>>{model_name} training complete in {training_time:.2f} seconds")
    torch.save(model.state_dict(), f'weights/{model_name}_final.pth')

    return history, training_time
