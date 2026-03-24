import numpy as np
from torch.utils.data import Dataset, DataLoader

class AnnotatedDataset(Dataset):
    def __init__(self, df, tokenizer, max_seq_len=2048):
        self.df = df.loc[:, ['prompt', 'response']].values
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        full_text = self.df[idx, 0] + self.df[idx, 1] + self.tokenizer.eos_token
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_len,
            return_offsets_mapping=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        prompt_end_char = len(self.df[idx, 0])
        labels = [
            -100 if start < prompt_end_char else token_id
            for (start, _), token_id in zip(offsets, input_ids)
        ]

        input_ids = np.array(input_ids)
        if np.random.randn() < 0.5:
            # Only do the masking with 50% probability
            mask = np.random.rand(input_ids.shape[0]) < 0.03
            input_ids[mask] = self.tokenizer.pad_token_type_id
        return {"input_ids": input_ids, "labels": labels}