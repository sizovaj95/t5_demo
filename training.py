import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime as dt
from rouge_score import rouge_scorer
from common import ReviewsDataset
import common as com


EPOCHS = 3
MODEL_NAME = "t5-base"
BATCH_SIZE = 8


def train(epochs: int, model, device, train_loader: DataLoader, optimizer, pad_token_id: int):
    model.train()

    for epoch in range(epochs):
        for train_data in tqdm(train_loader):

            input_ids = train_data['input_ids'].to(device)
            input_mask = train_data['input_mask'].to(device)

            target_ids = train_data['output_ids'].to(device)
            target_ids[target_ids == pad_token_id] = -100

            outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=target_ids)
            loss = outputs[0]
            print(f"Epoch {epoch+1} | Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def validate(model: T5ForConditionalGeneration, test_loader: DataLoader, tokenizer: T5Tokenizer, device):
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for test_data in test_loader:
            target_ids = test_data['output_ids'].to(device)
            input_ids = test_data['input_ids'].to(device)
            input_mask = test_data['input_mask'].to(device)

            predicted_ids = model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                max_length=com.MAX_OUTPUT_LEN
            )
            predictions_ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            for g in predicted_ids]
            targets_ = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                        target_ids]

            predictions.extend(predictions_)
            targets.extend(targets_)
    return predictions, targets


def get_max_token_numbers_from_training_data():
    with open("training_data.json", "r", encoding="utf8") as f:
        data = json.load(f)
    df = pd.DataFrame.from_records(data)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    review_lens = []
    summary_lens = []
    for i, row in df.iterrows():
        review = row['review']
        summary = row['summary']
        rev_len = len(tokenizer.encode(review, max_length=5000))
        sum_len = len(tokenizer.encode(summary, max_length=500))
        review_lens.append(rev_len)
        summary_lens.append(sum_len)
    rev_max = max(review_lens)
    sum_max = max(summary_lens)

    print(f"Longest review: {rev_max} tokens")
    print(f"Longest summary: {sum_max} tokens")
    print("======================================")


def save_model(model, train_df_len: int):
    today = dt.now()
    dt_now_str = f"{today.year}{today.month}{today.day}"
    model_name = f"{train_df_len}_{dt_now_str}_{com.MAX_OUTPUT_LEN}_{EPOCHS}.pt"
    try:
        print(f"Saving {model_name}")
        torch.save(model, "models//" + model_name)
    except Exception as er:
        print(f"Failed to save {model_name}: {er}")


def print_test_results(model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, test_data: ReviewsDataset, device):
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    predictions, targets = validate(model, test_loader, tokenizer, device)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for pred, tar in zip(predictions, targets):
        print(f"Target: {tar}\nPredicted: {pred}")
        scores = scorer.score(tar, pred)
        print(scores)
        print("\n")


def main():
    torch.manual_seed(com.RANDOM_SEED)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=com.MAX_INPUT_LEN)
    device = torch.device('cpu')

    lr = 1e-4

    train_df, test_df = com.load_and_maybe_split_data(com.TRAIN_FRACTION, com.RANDOM_SEED)

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    train_dataset = ReviewsDataset(train_df, tokenizer, com.MAX_INPUT_LEN, com.MAX_OUTPUT_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(EPOCHS, model, device, train_loader, optimizer, tokenizer.pad_token_id)
    save_model(model, len(train_df))

    # if test_df is not None:
    #     test_dataset = ReviewsDataset(test_df, tokenizer, com.MAX_INPUT_LEN, com.MAX_OUTPUT_LEN)
    #     print_test_results(model, tokenizer, test_dataset, device)


if __name__ == "__main__":
    start = time.time()
    # get_max_token_numbers_from_training_data()
    main()
    print("Time taken: {0:.2f} s".format(time.time() - start))
