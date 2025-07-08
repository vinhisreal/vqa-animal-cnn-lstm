import argparse
import torch
from PIL import Image
from torchvision import transforms

from src.data.utils import normalize_text
from src.models.cnn_lstm_seq2seq import CNN_LSTM_Seq2Seq
from src.models.resnet18_attention import ResNet18_LSTM_LuongAttention
from src.models.resnet50_attention import ResNet50_LSTM_LuongAttention
from src.models.densenet_attention import DenseNet121_LSTM_LuongAttention

# ===== Predict function =====
def predict(image_path, question, model, vocab, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    question = normalize_text(question)
    question_encoded = vocab.encode(question)
    question_tensor = torch.tensor(question_encoded).unsqueeze(0).to(device)
    
    answer_input = torch.tensor([[vocab.word2idx["<SOS>"]]]).to(device)
    generated = []

    with torch.no_grad():
        for _ in range(20):
            output = model(image, question_tensor, answer_input)
            next_token = output.argmax(2)[:, -1].item()
            if next_token == vocab.word2idx["<EOS>"]:
                break
            generated.append(next_token)
            answer_input = torch.cat([answer_input, torch.tensor([[next_token]]).to(device)], dim=1)

    return vocab.decode(generated)


# ===== Model loading =====
def load_model(model_name, vocab_size, device):
    if model_name == "cnn_lstm":
        model = CNN_LSTM_Seq2Seq(vocab_size=vocab_size)
    elif model_name == "resnet18":
        model = ResNet18_LSTM_LuongAttention(vocab_size=vocab_size)
    elif model_name == "resnet50":
        model = ResNet50_LSTM_LuongAttention(vocab_size=vocab_size)
    elif model_name == "densenet121":
        model = DenseNet121_LSTM_LuongAttention(vocab_size=vocab_size)
    else:
        raise ValueError(f"Model not supported: {model_name}")
    
    model.load_state_dict(torch.load(f"weights/{model_name}_final.pth", map_location=device))
    return model.to(device)


# ===== Main CLI =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA Animal Inference")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn ảnh")
    parser.add_argument("--question", type=str, required=True, help="Câu hỏi")
    parser.add_argument("--model", type=str, choices=["cnn_lstm", "resnet18", "resnet50", "densenet121"],
                        default="resnet18", help="Tên model để sử dụng")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load vocab
    from src.data.utils import load_data
    vocab, _, _ = load_data(batch_size=2)  

    # Load model
    model = load_model(args.model, vocab.vocab_size, args.device)

    # Predict
    answer = predict(args.image, args.question, model, vocab, args.device)
    print(f"Answer: {answer}")
