# run_training.py
import os
import torch
from data.utils import load_data
from models.cnn_lstm_seq2seq import CNN_LSTM_Seq2Seq
from models.resnet18_attention import ResNet18_LSTM_LuongAttention
from models.resnet50_attention import ResNet50_LSTM_LuongAttention
from models.densenet_attention import DenseNet121_LSTM_LuongAttention
from train import train_model, count_parameters
from evaluation import evaluate_on_testset, evaluate_model, compare_models, predict
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, train_loader, test_loader = load_data()

    _, test_dataset = train_loader.dataset, test_loader.dataset

    num_epochs = 20
    print("=" * 50)
    print("TRAINING")
    print("=" * 50)

    # CNN + LSTM
    print("\n--- CNN + LSTM ---")
    cnn_lstm_model = CNN_LSTM_Seq2Seq(vocab_size=vocab.vocab_size).to(device)
    cnn_lstm_history, cnn_lstm_time = train_model(cnn_lstm_model, "cnn_lstm", vocab, train_loader, test_loader, device, num_epochs)

    # ResNet18
    print("\n--- ResNet18 + LSTM + Luong ---")
    resnet18_model = ResNet18_LSTM_LuongAttention(vocab_size=vocab.vocab_size).to(device)
    resnet18_history, resnet18_time = train_model(resnet18_model, "resnet18", vocab, train_loader, test_loader, device, num_epochs)

    # ResNet50
    print("\n--- ResNet50 + LSTM + Luong ---")
    resnet50_model = ResNet50_LSTM_LuongAttention(vocab_size=vocab.vocab_size).to(device)
    resnet50_history, resnet50_time = train_model(resnet50_model, "resnet50", vocab, train_loader, test_loader, device, num_epochs)

    # DenseNet121
    print("\n--- DenseNet121 + LSTM + Luong ---")
    densenet121_model = DenseNet121_LSTM_LuongAttention(vocab_size=vocab.vocab_size).to(device)
    densenet121_history, densenet121_time = train_model(densenet121_model, "densenet121", vocab, train_loader, test_loader, device, num_epochs)

    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    cnn_bleu, cnn_rouge = evaluate_on_testset(cnn_lstm_model, test_loader, vocab, device)
    res18_bleu, res18_rouge = evaluate_on_testset(resnet18_model, test_loader, vocab, device)
    res50_bleu, res50_rouge = evaluate_on_testset(resnet50_model, test_loader, vocab, device)
    dense_bleu, dense_rouge = evaluate_on_testset(densenet121_model, test_loader, vocab, device)

    model_results = [
        {"name": "CNN+LSTM", "params": count_parameters(cnn_lstm_model), "train_time": cnn_lstm_time, "bleu": cnn_bleu, "rouge": cnn_rouge},
        {"name": "ResNet18", "params": count_parameters(resnet18_model), "train_time": resnet18_time, "bleu": res18_bleu, "rouge": res18_rouge},
        {"name": "ResNet50", "params": count_parameters(resnet50_model), "train_time": resnet50_time, "bleu": res50_bleu, "rouge": res50_rouge},
        {"name": "DenseNet121", "params": count_parameters(densenet121_model), "train_time": densenet121_time, "bleu": dense_bleu, "rouge": dense_rouge}
    ]

    compare_df = compare_models(model_results)
    compare_df.to_csv("model_comparison_results.csv", index=False)
    print("\n\u2714 Saved model_comparison_results.csv")

    print("\n\nDetails of each random model:")
    evaluate_model(cnn_lstm_model, "CNN_LSTM", test_dataset, vocab, device, top_k=5)
    evaluate_model(resnet18_model, "ResNet18", test_dataset, vocab, device, top_k=5)
    evaluate_model(resnet50_model, "ResNet50", test_dataset, vocab, device, top_k=5)
    evaluate_model(densenet121_model, "DenseNet121", test_dataset, vocab, device, top_k=5)

    sample_image = "path_to_test_image.jpg" 
    sample_question = "What animal is this?"

    if os.path.exists(sample_image):
        img = Image.open(sample_image)
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        for model, name in zip([cnn_lstm_model, resnet18_model, resnet50_model, densenet121_model],
                            ["CNN_LSTM", "ResNet18", "ResNet50", "DenseNet121"]):
            answer = predict(sample_image, sample_question, model, vocab, device)
            print(f"{name} Answer: {answer}")
