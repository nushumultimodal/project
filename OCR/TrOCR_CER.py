import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import unicodedata
from itertools import zip_longest
import numpy as np


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


df = pd.read_csv('_.csv') # Replace with actual csv file
df['file_name'] = df.index + 1
df['file_name'] = df['file_name'].apply(lambda x: f"nushu_{x}.png")


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['Nushu'][idx]
        image_path = os.path.join(self.root_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length
        ).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

# ------------------- tool functions -------------------
def is_printable_char(char):
    try:
        return char != '\ufffd' and not unicodedata.category(char).startswith("C")
    except:
        return False

def clean_text(text):
    return ''.join(c for c in text if is_printable_char(c))

def extract_correct_characters(pred_text, label_text):
    return [p for p, l in zip_longest(pred_text, label_text, fillvalue='') if p == l and is_printable_char(p)]

def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

# ------------------- evaluate function -------------------
def evaluate_model_accuracy(model, processor, dataset, max_samples=None):
    model.eval()
    model.to(device)

    predictions, references = [], []
    exact_matches = 0
    correct_char_count = 0
    total_char_count = 0
    cer_total = 0
    ref_char_total = 0

    total = min(len(dataset), max_samples) if max_samples else len(dataset)

    with torch.no_grad():
        for i in tqdm(range(total), desc="Evaluating"):
            sample = dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            labels = sample["labels"]

            # decode ground truth and predict text
            label_ids = [t if t != -100 else processor.tokenizer.pad_token_id for t in labels.tolist()]
            raw_label_text = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
            raw_pred_text = processor.tokenizer.decode(
                model.generate(pixel_values)[0],
                skip_special_tokens=True
            )

            label_text = clean_text(raw_label_text)
            pred_text = clean_text(raw_pred_text)

            predictions.append(pred_text)
            references.append(label_text)

            if pred_text.strip() == label_text.strip():
                exact_matches += 1

            correct_chars = extract_correct_characters(pred_text, label_text)
            correct_char_count += len(correct_chars)
            total_char_count += len(label_text)

            if len(label_text) > 0:
                dist = levenshtein(pred_text, label_text)
                cer_total += dist
                ref_char_total += len(label_text)

                if dist > len(label_text):
                    print(f"⚠️ CER > 1 at sample {i+1}: Distance={dist}, Ref length={len(label_text)}")
            else:
                print(f"⚠️ Skipped empty reference at sample {i+1}")

            if i < 100:
                correct_str = ''.join(correct_chars)
                print(f"\nSample {i+1}")
                print(f"▶ Ground Truth : {label_text}")
                print(f"▶ Prediction   : {pred_text}")
                print(f"✅ Correct non-gibberish characters ({len(correct_chars)} / {len(label_text)}): {correct_str}")

    cer = cer_total / ref_char_total if ref_char_total else 0
    acc = exact_matches / len(predictions)
    char_level_acc = correct_char_count / total_char_count if total_char_count else 0

    print(f"\n🧠 Character Error Rate (CER): {cer:.4f}")
    print(f"🎯 Exact Match Accuracy      : {acc:.4f}")
    print(f"✅ Character-level Accuracy  : {char_level_acc:.4f}")

    return cer, acc, char_level_acc

# ------------------- prepare data -------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df.reset_index(drop=True, inplace=True)
eval_dataset = IAMDataset(root_dir='nvshu_png_images/', df=test_df, processor=processor)

# ------------------- load model and evaluate -------------------
checkpoint_path = "./model_save/checkpoint-1400"
model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
model.to(device)

cer, acc, char_acc = evaluate_model_accuracy(model, processor, eval_dataset)

print("✅ Final CER:", cer)
print("✅ Final Accuracy:", acc)
print("✅ Final Character-level Accuracy:", char_acc)
