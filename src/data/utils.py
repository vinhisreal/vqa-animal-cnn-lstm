# src/data/utils.py

import json
import os
import re
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DEFAULT_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_annotations.json")
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "animals")

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Class Vocabulary
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_count = Counter()
        self.vocab_size = 4

    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence.lower().split():
                self.word_count[word] += 1
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, sentence):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence.lower().split()]

    def decode(self, tokens):
        return " ".join([self.idx2word[token] for token in tokens if token not in {0, 1, 2}])

# Custom Dataset
class ImageQuestionDataset(Dataset):
    def __init__(self, json_data, image_dir, vocab, transform=None):
        self.data = json_data
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.samples = []

        self.class_map = {
            "0": "cats",
            "1": "dogs",
            "2": "snakes"
        }

        for image_name, item in self.data.items():
            class_id = image_name.split("_")[0]
            class_name = self.class_map.get(class_id)

            if class_name is None:
                print(f"Undefine: {image_name}")
                continue

            img_path = os.path.join(self.image_dir, class_name, image_name)

            if not os.path.exists(img_path):
                print(f"Image can't found: {img_path}")
                continue

            for q, a in zip(item["questions"], item["answers"]):
                self.samples.append((img_path, q, a))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, question, answer = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        question = normalize_text(question)
        answer = normalize_text(answer)

        question_encoded = self.vocab.encode(question)
        answer_encoded = [self.vocab.word2idx["<SOS>"]] + self.vocab.encode(answer) + [self.vocab.word2idx["<EOS>"]]

        return image, torch.tensor(question_encoded), torch.tensor(answer_encoded)

# Collate Function
def collate_fn_seq2seq(batch):
    images, questions, answers = zip(*batch)
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=0)
    answers_padded = pad_sequence(answers, batch_first=True, padding_value=0)
    return torch.stack(images), questions_padded, answers_padded

# Tiện ích lấy class từ tên ảnh
def get_class_from_filename(filename):
    return filename.split("_")[0]

# Hàm load dữ liệu

def load_data(
    json_path=DEFAULT_JSON_PATH,
    image_dir=DEFAULT_IMAGE_DIR,
    batch_size=32,
):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    vocab = Vocabulary()
    all_questions = [normalize_text(q) for item in data.values() for q in item["questions"]]
    vocab.build_vocab(all_questions)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ImageQuestionDataset(data, image_dir, vocab, transform=train_transforms)
    labels = [get_class_from_filename(os.path.basename(sample[0])) for sample in dataset.samples]

    train_samples, test_samples = train_test_split(
        dataset.samples,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_dataset = ImageQuestionDataset(data, image_dir, vocab, transform=train_transforms)
    train_dataset.samples = train_samples

    test_dataset = ImageQuestionDataset(data, image_dir, vocab, transform=test_transforms)
    test_dataset.samples = test_samples

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_seq2seq)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq2seq)

    return vocab, train_loader, test_loader
