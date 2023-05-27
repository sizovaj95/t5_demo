from torch.utils.data import Dataset
import pandas as pd
import json
from sklearn.model_selection import train_test_split


MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 128
RANDOM_SEED = 42
TRAIN_FRACTION = 0.8


class ReviewsDataset(Dataset):
    def __init__(self, reviews: pd.DataFrame, tokenizer, max_input_len: int, max_output_len: int):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> dict:
        review_text = self.reviews.iloc[idx]['review']
        review = "summarize: " + review_text
        summary = self.reviews.iloc[idx]['summary']

        encoded_input = self.tokenizer([review], max_length=self.max_input_len, pad_to_max_length=True,
                                       truncation=True,
                                       padding="max_length", return_tensors="pt")
        encoded_output = self.tokenizer([summary], max_length=self.max_output_len, pad_to_max_length=True,
                                        truncation=True,
                                        padding="max_length", return_tensors="pt")
        input_ids = encoded_input['input_ids'].squeeze()
        input_mask = encoded_input['attention_mask'].squeeze()
        output_ids = encoded_output['input_ids'].squeeze()

        return {
            "review_text": review_text,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "output_ids": output_ids
        }


def load_and_maybe_split_data(train_faction: float, random_seed: int, split: bool = True) -> tuple:
    with open("training_data.json", "r", encoding="utf8") as f:
        data = json.load(f)
    df = pd.DataFrame.from_records(data)
    if split:
        train_df, test_df = train_test_split(df, train_size=train_faction, random_state=random_seed)
    else:
        train_df, test_df = df, None
    return train_df, test_df
