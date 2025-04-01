import torch
from torch.utils.data import Dataset


class EmotionsDataset(Dataset):

    def __init__(self, content, labels, tokenizer, max_length):
        self.content = content
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        tweet = str(self.content[idx])
        label = self.labels[idx]

        encoded_sent = self.tokenizer(tweet, padding='max_length',
                                      max_length=self.max_length)

        o = dict()
        o['input_ids'] = encoded_sent['input_ids']
        o['attention_mask'] = encoded_sent['attention_mask']
        o['labels']= torch.tensor(label)
        return o

    def __len__(self):

        return len(self.content)
