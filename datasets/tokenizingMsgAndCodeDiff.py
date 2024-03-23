import numpy as np
import torch

class tokenizeMsgAndCodeDiff():
    def __init__(self, max_input_length, tokenizer):
            self.max_input_length = max_input_length
            self.tokenizer = tokenizer

    def TokenizeMsgAndCodeDiff(self, MsgAndCodeDiff):
        msg, codediff = MsgAndCodeDiff["msg"].values, MsgAndCodeDiff["OriginalCodeDiff"].values
        if len(codediff) > 1:
            print()
        nlnl_tokenizer = self.tokenize_nlnl(msg)
        nlpl_tokenizer = self.tokenize_nlpl(codediff)

        padding_length = self.max_input_length - nlnl_tokenizer['input_ids'].shape[1] - \
                         nlpl_tokenizer['input_ids'].shape[1] + 1  # '+1' for [CLS] of pr_tokens

        paddings_tokens: dict = {
            'input_ids': torch.full((padding_length,), self.tokenizer.pad_token_id),
            'attention_mask': torch.full((padding_length,), 0),
        }
        res_tokens = {
            'input_ids': torch.cat(
                (
                    nlnl_tokenizer['input_ids'][0, :],
                    nlpl_tokenizer['input_ids'][0, 1:], paddings_tokens['input_ids'])
            ),  # [1:] to remove [CLS] of pr_tokens
            'attention_mask': torch.cat(
                (
                    nlnl_tokenizer['attention_mask'][0, :],
                    nlpl_tokenizer['attention_mask'][0, 1:], paddings_tokens['attention_mask'])
            ),
        }
        res_tokens["input_ids"] = res_tokens["input_ids"].unsqueeze(0)
        res_tokens["attention_mask"] = res_tokens["attention_mask"].unsqueeze(0)
        return res_tokens

    def tokenize_nlpl(self, codediff):

        flat_list = [item for sublist in codediff for item in sublist]
        codediff = ",".join(flat_list[0])
        codediff_tokens = self.tokenizer(
            codediff,
            max_length=self.max_input_length//2,
            padding=False,
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True
        )

        return codediff_tokens

    def tokenize_nlnl(self,  msg):
        msg_array = [item for sublist in msg for item in sublist]
        msg = ",".join(msg_array)
        msg_tokens = self.tokenizer(
            msg,
            max_length=self.max_input_length//2 + 1,
            padding=False,
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True
        )

        return msg_tokens