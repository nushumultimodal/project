import os
import pandas as pd
import unicodedata
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import zip_longest
from sklearn.model_selection import train_test_split
from google.cloud import vision
import io
import os

# ------------------- Google Cloud Auth -------------------
# Set environment variable GOOGLE_APPLICATION_CREDENTIALS to indicate JSON keys
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_cloud_vision/xxxxxx.json" # replace your own json file

# ------------------- Path Configuration -------------------
IMG_DIR = '_/'
CSV_PATH = '_.csv'

# ------------------- Load CSV -------------------
df = pd.read_csv(CSV_PATH)
df['file_name'] = df.index + 1
df['file_name'] = df['file_name'].apply(lambda x: f"nushu_{x}.png")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df.reset_index(drop=True, inplace=True)

# ------------------- Utility Functions -------------------
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

# ------------------- Google Cloud Vision Evaluation -------------------
def evaluate_google_vision_accuracy(test_df, img_dir, max_samples=None):
    client = vision.ImageAnnotatorClient()
    
    predictions, references = [], []
    exact_matches = 0
    correct_char_count = 0
    total_char_count = 0
    cer_total = 0
    ref_char_total = 0

    total = min(len(test_df), max_samples) if max_samples else len(test_df)

    for i in tqdm(range(total), desc="Evaluating (Google Vision OCR)"):
        row = test_df.iloc[i]
        image_path = os.path.join(img_dir, row["file_name"])
        gt_text = clean_text(str(row["Nushu"]))

        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            # image_context = vision.ImageContext(language_hints=["zh-Hans"])  # simplified chinese
            image_context = vision.ImageContext(language_hints=["zh-Hant"])  # traditional chinese
            response = client.document_text_detection(image=image)
            pred_text = response.full_text_annotation.text.strip()
        except Exception as e:
            print(f"❌ Google Vision OCR failed on {row['file_name']}: {e}")
            pred_text = ""

        pred_text = clean_text(pred_text)

        predictions.append(pred_text)
        references.append(gt_text)

        if pred_text.strip() == gt_text.strip():
            exact_matches += 1

        correct_chars = extract_correct_characters(pred_text, gt_text)
        correct_char_count += len(correct_chars)
        total_char_count += len(gt_text)

        if len(gt_text) > 0:
            dist = levenshtein(pred_text, gt_text)
            cer_total += dist
            ref_char_total += len(gt_text)
        else:
            print(f"⚠️ Skipped empty reference at sample {i+1}")

        if i < 100:
            correct_str = ''.join(correct_chars)
            print(f"\nSample {i+1}")
            print(f"▶ Ground Truth : {gt_text}")
            print(f"▶ Prediction   : {pred_text}")
            print(f"✅ Correct characters ({len(correct_chars)} / {len(gt_text)}): {correct_str}")

    cer = cer_total / ref_char_total if ref_char_total else 0
    acc = exact_matches / len(predictions)
    char_level_acc = correct_char_count / total_char_count if total_char_count else 0

    print(f"\n🧠 [GoogleOCR] Character Error Rate (CER): {cer:.4f}")
    print(f"🎯 [GoogleOCR] Exact Match Accuracy      : {acc:.4f}")
    print(f"✅ [GoogleOCR] Character-level Accuracy  : {char_level_acc:.4f}")

    return cer, acc, char_level_acc

# ------------------- Run Evaluation -------------------
cer, acc, char_acc = evaluate_google_vision_accuracy(test_df, IMG_DIR)
print("✅ Final CER:", cer)
print("✅ Final Accuracy:", acc)
print("✅ Final Character-level Accuracy:", char_acc)
