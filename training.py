import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time


t5_model = "t5-base"
max_input_len = 1024
max_output_len = 50
random_seed = 42


class ReviewsDataset(Dataset):
    def __init__(self, reviews: pd.DataFrame, tokenizer):
        self.reviews = reviews
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> dict:
        review = "summarize: " + self.reviews.iloc[idx]['review']
        summary = self.reviews.iloc[idx]['summary']

        encoded_input = self.tokenizer([review], max_length=max_input_len, pad_to_max_length=True, truncation=True,
                                       padding="max_length", return_tensors="pt")
        encoded_output = self.tokenizer([summary], max_length=max_output_len, pad_to_max_length=True, truncation=True,
                                        padding="max_length", return_tensors="pt")
        input_ids = encoded_input['input_ids'].squeeze()
        input_mask = encoded_input['attention_mask'].squeeze()
        output_ids = encoded_output['input_ids'].squeeze()

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "output_ids": output_ids
        }


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
    texts = []
    with torch.no_grad():
        for test_data in test_loader:
            target_ids = test_data['output_ids'].to(device)
            input_ids = test_data['input_ids'].to(device)
            input_mask = test_data['input_mask'].to(device)

            predicted_ids = model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                max_length=max_output_len
            )
            predictions_ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            for g in predicted_ids]
            targets_ = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                        target_ids]
            texts_ = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                      input_ids]

            predictions.extend(predictions_)
            targets.extend(targets_)
            texts.extend(texts_)
    return predictions, targets, texts


def main():
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    tokenizer = T5Tokenizer.from_pretrained(t5_model, model_max_length=max_input_len)

    with open("training_data.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame.from_records(data)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    epochs = 3
    batch_size = 8
    lr = 1e-4
    train_frac = 0.8

    train_df, test_df = train_test_split(df, train_size=train_frac, random_state=random_seed)

    model = T5ForConditionalGeneration.from_pretrained(t5_model).to(device)
    train_dataset = ReviewsDataset(train_df, tokenizer)
    test_dataset = ReviewsDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(epochs, model, device, train_loader, optimizer, tokenizer.pad_token_id)
    try:
        torch.save(model, f"{len(df)}_model_w_prefix_{max_output_len}_output.pt")
        # torch.save(model.state_dict(), "first_model_state_dict_w_prefix.pth")
    except:
        print(":(")
    # model.save_pretrained()
    # tokenizer.save_pretrained()

    # model = torch.load("first_model_w_prefix.pt")

    predictions, targets, texts = validate(model, test_loader, tokenizer, device)
    for pred, tar in zip(predictions, targets):
        print(f"Target: {tar}\nPredicted: {pred}")
        print("\n")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Time taken: {time.time() - start}")
