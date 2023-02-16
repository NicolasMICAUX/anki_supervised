"""
Train a transformer on Anki dataset, with dependency parsing as features.
"""
# !pip install datasets transformers accelerate
# from huggingface_hub import notebook_login
# notebook_login()  # READ ACCESS : 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
# huggingface-cli login to login

# *** PROCESS RAW DATA TO CREATE THE DEPENDENCY PARSING DATASET ***
from datasets import load_dataset, load_metric
import re
from supar import Parser


dataset = load_dataset("nicolasmicaux/anki_raw_data", use_auth_token=True)
parser = Parser.load('biaffine-dep-xlmr')  # multi-lingual dependency parser
# todo : pleins de modèles différents dans les assets de https://github.com/yzhangcs/parser/releases/tag/v1.1.0


# if the gpu device is available
# torch.cuda.set_device('cuda:0')


# CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
IS_CLOZE_NB = 0
TOTAL = 0


def process(row):
    global CLEANR, IS_CLOZE_NB, TOTAL
    text = row['field']
    # Preprocess text - delete all html tags => already done prior to this
    # text = re.sub(CLEANR, '', text)  # https://stackoverflow.com/a/12982689

    chunks = []
    for tmp in text.split('}}'):
        chunks.extend(re.split(r'{{c[0-9]+::', tmp))

    # hacky ! on tire profit du fait que tokenizer(List[str]) re-tokenize chaque str

    ids = parser.predict(chunks, lang='en', verbose=True)
    # self, data, pred=None, lang=None, buckets=8, batch_size=5000, prob=False, **kwargs

    starts_with_cloze = int(text.startswith('{{'))

    is_cloze = [0 if i is None else (i + starts_with_cloze) % 2 for i in ids.word_ids()]

    ids['labels'] = is_cloze  # name of the labels is often "labels" in the fine-tuned models below

    # assert len(is_cloze) == len(ids['input_ids']) (always true)

    IS_CLOZE_NB += sum(is_cloze)
    TOTAL += len(is_cloze)
    return ids


tokenized_dataset = dataset.map(process)
print('Proportion IS_CLOZE : ', IS_CLOZE_NB / TOTAL)

# *** Export ***
# tokenized_dataset['train'] = tokenized_dataset['train'].remove_columns(['field', 'file', 'usn', 'guid', 'mid', 'mod', 'id', 'tags', 'lang'])
# tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1)
# Push dataset to Huggingface Hub
# notebook_login()  # WRITE ACCESS : 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
# tokenized_dataset.push_to_hub("nicolasmicaux/anki_data", private=True)

exit()

# Use GPU 1
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# W&B
# !pip install wandb
import wandb

wandb.login()  # 2fc5ae6c733ca9b3eaa6e40ab0097ac45364a294

from huggingface_hub import get_full_repo_name

model_name = f"dep-parse-anki"
output_dir = model_name
repo_name = get_full_repo_name(model_name)

# from transformers import DataCollatorForTokenClassification
# data_collator = DataCollatorForTokenClassification(tokenizer)

# accelerator to use GPU
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_dataset["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_dataset['test'], collate_fn=data_collator, batch_size=batch_size
)

from torch.optim import AdamW

lr = 2e-5
optimizer = AdamW(model.parameters(), lr=lr)

from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

accuracy = load_metric("accuracy")
precision = load_metric("precision")
recall = load_metric("recall")
f1 = load_metric("f1")

label_names = [0, 1]  # is_cloze


def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l in label_names] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l in label_names]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


from tqdm.auto import tqdm
import torch

config = {
    "learning_rate": lr,
    "batch_size": batch_size,
    'base_model': model_checkpoint,
}

wandb.init(reinit=True, config=config, project='anki', job_type='train', tags=['supervised'], group='supervised')
# name, notes, magic
progress_bar = tqdm(range(num_training_steps))

# criterion = torch.nn.CrossEntropyLoss(weights=class_weights)
for epoch in range(num_train_epochs):
    # Training
    model.train()
    for i, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        progress_bar.update(1)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        true_predictions, true_labels = postprocess(predictions, labels)
        for pred, lab in zip(true_predictions, true_labels):
            accuracy.add_batch(predictions=pred, references=lab)
            precision.add_batch(predictions=pred, references=lab)
            recall.add_batch(predictions=pred, references=lab)
            f1.add_batch(predictions=pred, references=lab)

        # log metrics
        if i % 100 == 0:
            results = [accuracy.compute(), precision.compute(), recall.compute(), f1.compute()]
            wandb.log({'epoch': epoch, 'accuracy': results[0], 'precision': results[1], 'recall': results[2],
                       'f1': results[3]}, commit=False)
        wandb.log({'loss': loss, 'lr': lr_scheduler.get_last_lr()[0]})

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        true_predictions, true_labels = postprocess(predictions, labels)
        for pred, lab in zip(true_predictions, true_labels):
            accuracy.add_batch(predictions=pred, references=lab)
            precision.add_batch(predictions=pred, references=lab)
            recall.add_batch(predictions=pred, references=lab)
            f1.add_batch(predictions=pred, references=lab)

    results = [accuracy.compute(), precision.compute(), recall.compute(), f1.compute()]
    print(f"epoch {epoch}:", results)
    wandb.log({'epoch': epoch, 'accuracy': results[0], 'precision': results[1], 'recall': results[2], 'f1': results[3]})

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

unwrapped_model.push_to_hub(repo_name)

tokenizer.push_to_hub(repo_name)

wandb.finish()
