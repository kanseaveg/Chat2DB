import os
import json
import torch

from torch.utils.data import DataLoader

from modelend.utils.load_dataset import Text2SQLDataset
from modelend.utils.text2sql_decoding_utils import decode_natsqls
import logging
logging.basicConfig(level=logging.INFO)
def text2sql(dataset, opt, tokenizer, model, tablenat):
    dev_dataset = Text2SQLDataset(
        dataset=dataset,
        mode=opt.mode
    )

    dev_dataloder = DataLoader(
        dev_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        drop_last=False
    )
    for batch in dev_dataloder:
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )

        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids=encoder_input_ids,
                attention_mask=encoder_input_attention_mask,
                max_length=256,
                decoder_start_token_id=model.config.decoder_start_token_id,
                num_beams=opt.num_beams,
                num_return_sequences=opt.num_return_sequences
            )
            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])

            predict_sql = decode_natsqls(
                opt,
                model_outputs,
                batch_db_ids,
                tokenizer,
                batch_tc_original,
                tablenat
            )

    return predict_sql