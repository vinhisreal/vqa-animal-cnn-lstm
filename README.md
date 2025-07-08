# VQA-Animal: Visual Question Answering on Animal Images
This project is a Visual Question Answering (VQA) system focused on animal images (cats, dogs, snakes). Given an image and a natural language question about it, the model generates a natural language answer.

The system supports multiple CNN backbones and uses an LSTM decoder (with Luong Attention in some variants). It is implemented in PyTorch and can be trained and evaluated easily either locally or on platforms like Kaggle.

<p align="center">
  <img src="https://res.cloudinary.com/vinhisreal/image/upload/v1751709714/Screenshot_2025-07-05_170031_ip2mn9.png" width="500"/>
</p>

## Installation
### Clone the repository:
    ```bash
    git clone https://github.com/vinhisreal/vqa-animal-cnn-lstm.git
    cd vqa-animal-cnn-lstm
    ```
### Install Dependencies
You can install all required libraries using requirements.txt:
    ```bash
    pip install -r requirements.txt
    ```
### Alternatively, install them manually:
    ```bash
    pip install torch torchvision scikit-learn nltk pandas matplotlib pillow rouge-score
    ```

## Dataset Format
The cleaned_annotations.json file contains image names mapped to questions and answers:
    ```json
    "0_0001.jpg": {
        "description": "A black and brown cat being held by its owner",
        "questions": [
            "What type of animal is being held?",
            "How many animals are there?",
            "What color is the cat?"
        ],
        "answers": [
            "A cat.",
            "One.",
            "Black and brown."
        ]
    }
    ```
Prefix 0_ means the image is in cats/

Prefix 1_ means the image is in dogs/

Prefix 2_ means the image is in snakes/

## Available Models
**CNN + LSTM**

**ResNet18 + LSTM + Luong Attention**

**ResNet50 + LSTM + Luong Attention**

**DenseNet121 + LSTM + Luong Attention**

All models use the same vocabulary and dataset format.

## How to Use
### Train All Models
Run the following script to train all four models:
    ```bash
    python src/run_training.py
    ```
This script will:

Load data

Train four different models

Save checkpoints into the models/ directory

Evaluate and save results to model_comparison_results.csv

### Make Predictions
Use the CLI script to generate predictions from a given image and question:

    ```bash
    python src/predict.py \
      --image path/to/image.jpg \
      --question "What animal is this?" \
      --model resnet50
    ```
Optional arguments:

--model: One of cnn_lstm, resnet18, resnet50, densenet121

--device: cpu or cuda

## Evaluation
## 🔍 Model Comparison Summary

| Model        | Parameters  | Training Time (s) | BLEU Score | ROUGE-L |
|--------------|-------------|-------------------|------------|----------|
| CNN + LSTM   | 10,253,552  | 152               | 0.320      | 0.507    |
| ResNet18     | 12,789,001  | 161               | 0.341      | 0.533    |
| ResNet50     | 23,648,920  | 183               | 0.358      | 0.551    |
| DenseNet121  | 14,301,002  | 177               | 0.372      | 0.567    |

_See visual result:_

![Model Comparison](https://res.cloudinary.com/vinhisreal/image/upload/v1751709334/score_f01wjs.png)
![Model Comparison](https://res.cloudinary.com/vinhisreal/image/upload/v1751709334/time_rwcctg.png)

### Automatic Metrics
After training, the model is evaluated using:

BLEU Score (from NLTK)

ROUGE-L Score (from rouge_score)

Results are printed and saved in a CSV file.

## Training & Evaluation Results
### Training Loss Curve
DenseNet121 converges the fastest.

CNN+LSTM has slower convergence and higher loss.

### Training Time Comparison
CNN+LSTM trains fastest due to fewer parameters.

DenseNet121 and ResNet50 are heavier and take longer.

### Evaluation (BLEU & ROUGE-L Scores)
ResNet18 achieves the best BLEU and ROUGE-L scores.

CNN+LSTM underperforms across all metrics.

## Conclusion
In this project, we developed a complete Visual Question Answering (VQA) system for animal image datasets using various CNN backbones combined with LSTM-based decoders. Through systematic training and evaluation on a custom dataset of cats, dogs, and snakes, we gained several key insights:

RestNet achieved the best overall performance in terms of both BLEU and ROUGE-L scores, suggesting strong representation power and effective decoding.

ResNet50 also performed well, with ResNet18 offering a good trade-off between accuracy and speed.

CNN+LSTM, while computationally lighter, showed lower performance compared to deeper pretrained backbones.

When to Use Which Model
For fast training or low-resource environments: Choose CNN+LSTM

For a balance between performance and speed: Use ResNet18

This project demonstrates the effectiveness of combining convolutional feature extractors with attention-based sequence models in answering natural language questions about images. It also serves as a template for extending VQA tasks to other domains beyond animals.

## License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this code for personal or commercial purposes, as long as proper credit is given.

## Author
Developed by Wzinh
GitHub: vinhisreal
Contact: vinhquang2610345@gmail.com
