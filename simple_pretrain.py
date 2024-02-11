import argparse
import random
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from modeling_bartx import BartxForConditionalGeneration

#pretrain a model on causal LM    
class Trainer:
    def __init__(self, args):
        self.args = args
        raw_datasets = load_dataset(args.dataset_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
        self.train_data = raw_datasets["train"]
        self.validation_data = raw_datasets["validation"]
        
        self.model_checkpoint = args.checkpoint_path
        self.model_path = args.output_path
        self.max_input_length = args.max_input_length
        self.max_target_length = args.max_target_length
        self.batch_size = args.batch_size
        self.learning_rate=args.learning_rate

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = BartxForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2")

    # def preprocess_function(self, examples):
    #     # Split each text into two halves
    #     first_half_texts = []
    #     second_half_texts = []
    #     for doc in examples["text"]:
    #         split_point = len(doc) // 2  # Calculate the midpoint of the text
    #         first_half_texts.append(doc[:split_point])
    #         second_half_texts.append(doc[split_point:])
            
    #     model_inputs = self.tokenizer(
    #         first_half_texts, max_length=self.max_input_length, truncation=True)

    #     # Setup the tokenizer for targets
    #     with self.tokenizer.as_target_tokenizer():
    #         labels = self.tokenizer(
    #             second_half_texts, max_length=self.max_target_length, truncation=True)

    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs
        
    def preprocess_function(self, examples):
        processed_texts = []
        target_texts = []  # Renamed for clarity
        for doc in examples["text"]:
            if random.random() < 0.5:  # 50% chance to split as before
                split_point = len(doc) // 2
                processed_texts.append("Continue: "+doc[:split_point])
                target_texts.append(doc[split_point:])
            else:  # 50% chance to mask a random part
                mask_length = random.randint(1, max(1, len(doc) // 4))  # Determine length of the mask
                start_index = random.randint(0, max(0, len(doc) - mask_length - 1))
                # Save the masked part for the target
                masked_part = doc[start_index:start_index + mask_length]
                # Create the masked text
                masked_text = doc[:start_index] + "<mask>" + doc[start_index + mask_length:]
                processed_texts.append("Preencha: "+masked_text)
                target_texts.append(masked_part)  # Only the masked part is the target

        model_inputs = self.tokenizer(
            processed_texts, max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_texts, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def save(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

    def get_trainer(self):
    
        tokenized_datasets_train = self.train_data.map(
            self.preprocess_function,
            batched=True,
            num_proc=8,
            batch_size=32,
            load_from_cache_file=True
            )
        tokenized_datasets_val = self.validation_data.map(
            self.preprocess_function,
            batched=True,
            num_proc=8,
            batch_size=32,
            load_from_cache_file=True
            )


        args = Seq2SeqTrainingArguments(
            output_dir=self.model_path,
            evaluation_strategy="steps",
            eval_steps=1000,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=32,
            weight_decay=0.01,
            save_total_limit=1,
            save_steps=100,
            logging_steps=1,
            num_train_epochs=1,
            predict_with_generate=True,
            bf16=True,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model
        )

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets_train,
            eval_dataset=tokenized_datasets_val,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--dataset_name', type=str, default="datalawyer/legal-docs", #train_gl_fb_all_new_class.csv
                        help="path to train file")
    parser.add_argument('--checkpoint_path', type=str, default="models/Bartx", #./base/
                        help="path or name of checkpoint")
    parser.add_argument('--output_path', type=str, default="models/bartx_pt_docs_1.0",
                        help="path or name of output")
    parser.add_argument('--validation_split_percentage', type=int, default=5,
                        help="percentage of validation split")
    parser.add_argument('--max_input_length', type=int, default=8192,
                        help="max number of tokens in input")
    parser.add_argument('--max_target_length', type=int, default=8192,
                        help="max number of tokens in output")
    parser.add_argument('--batch_size', type=int, default=1, #48
                        help="batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, #48
                        help="learning rate")

    args = parser.parse_args()

    print("training")
    trainer = Trainer(args)

    try:
        trainer.get_trainer().train()
    except KeyboardInterrupt:
        print('Interrupted')

    print('Saving model ...')
    trainer.save()

