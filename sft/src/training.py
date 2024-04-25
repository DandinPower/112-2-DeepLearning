from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from transformers import HfArgumentParser, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)


@dataclass
class DataArguments:
    dataset_name_or_path: str = field(default=None)
    train_split: str = field(default="train")
    val_split: str = field(default="validation")


@dataclass
class LoraArguments:
    lora_alpha: int = field(default=None)
    lora_dropout: float = field(default=None)
    lora_rank: int = field(default=None)
    bias: str = field(default="none")


@dataclass
class MyTrainingArguments(TrainingArguments):
    max_seq_length: int = field(default=None)
    load_best_model_at_end: bool = field(default=True)
    language: str = field(default="zh")
    tags: str = field(default="nycu-112-2-deeplearning-hw2")


def get_lora_config(lora_args: LoraArguments) -> LoraConfig:
    return LoraConfig(
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        r=lora_args.lora_rank,
        bias=lora_args.bias,
        task_type="CAUSAL_LM"
    )


def show_tokenizer_config(tokenizer):
    print(f"Padding side: {tokenizer.padding_side}")
    print(f"Padding token: {tokenizer.pad_token}")
    print(f"Eos token: {tokenizer.eos_token}")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, MyTrainingArguments, LoraArguments))

    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    lora_config = get_lora_config(lora_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path)

    show_tokenizer_config(tokenizer)

    # # add new pad token to avoid confict with existing end of sentence token
    # tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # tokenizer.pad_token = "<PAD>"
    # tokenizer.padding_side = "right"

    # show_tokenizer_config(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
    )

    from peft import get_peft_model

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset(
        data_args.dataset_name_or_path, split=data_args.train_split)

    val_dataset = load_dataset(
        data_args.dataset_name_or_path, split=data_args.val_split)

    def preprocess_function(examples):
        return tokenizer(examples["text"])

    train_dataset = train_dataset.map(preprocess_function, batched=True)

    val_dataset = val_dataset.map(preprocess_function, batched=True)

    print(train_dataset[0])

    # response_template = "[/INST] "

    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template, tokenizer=tokenizer)

    # def formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example['text'])):
    #         output_texts.append(example['text'][i])
    #     return output_texts

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        peft_config=lora_config,
        max_seq_length=training_args.max_seq_length,
        dataset_text_field="text",
        # formatting_func=formatting_prompts_func,
        # data_collator=collator,
    )

    trainer.train()

    kwargs = {
        "language": training_args.language,
        "finetuned_from": model_args.model_name_or_path,
        "tags": training_args.tags,
        "dataset_tags": data_args.dataset_name_or_path,
        "dataset": data_args.dataset_name_or_path,
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
