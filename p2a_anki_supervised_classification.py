# -*- coding: utf-8 -*-
"""P2A - Anki supervised classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1behI6aI4HrBG3LB6UmI2U8X2YCm45j1J

**Token classification** : inspiré de [ce notebook](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb).  
**Multi-lingual** : inspiré de [ce notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/multilingual.ipynb#scrollTo=oIlK6-Fsd9EM).  
**GPU acceleration** : inspiré de [ce notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/chapter7/section2_pt.ipynb#scrollTo=OwupK09l6-3S).  
**Weighted loss** : inspiré de [ce lien](https://discuss.huggingface.co/t/class-weights-for-bertforsequenceclassification/1674/7).

*Ça s'appelle "token classification" :  les classiques sont :
NER (Named-entity recognition) Classify the entities in the text (person, organization, location...).
POS (Part-of-speech tagging) Grammatically classify the tokens (noun, verb, adjective...)
Chunk (Chunking) Grammatically classify the tokens and group them into "chunks" that go together*
régler le batch size pour éviter out of memory
todo : tester divers modèles

Redémarrer l'environnement suffit (pas besoin de réinitialiser)

# Setup & load dataset
"""
#
# !pip install datasets transformers accelerate
# !apt install git-lfs
# !git config --global credential.helper store
# To run the training on TPU, you will need to uncomment the followin line:
# !pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

# from huggingface_hub import notebook_login
# notebook_login()  # READ ACCESS : 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

from datasets import load_dataset, load_metric
tokenized_dataset = load_dataset("nicolasmicaux/anki_data", use_auth_token=True)

task = "chunk" # Should be one of "ner", "pos" or "chunk" : je pense chunk marchera le mieux si je pars d'un truc pretrained
model_checkpoint = 'xlm-roberta-base'
batch_size = 8

# MULTI-LINGUAL WITHOUT EMBEDDING
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # automatically choose Tokenizer

"""## Weights & Biases"""

# Commented out IPython magic to ensure Python compatibility.
# !pip install wandb
import wandb
wandb.login()  # 2fc5ae6c733ca9b3eaa6e40ab0097ac45364a294
# %env WANDB_PROJECT=anki

"""# Process raw data (Anki raw dataset)"""

# from datasets import load_dataset, load_metric
# dataset = load_dataset("nicolasmicaux/anki_raw_data", use_auth_token=True)
#
# import re
# # CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
# IS_CLOZE_NB = 0
# TOTAL = 0
#
# def process(row):
#   global CLEANR, IS_CLOZE_NB, TOTAL
#   text = row['field']
#   # Preprocess text - delete all html tags => already done prior to this
#   # text = re.sub(CLEANR, '', text)  # https://stackoverflow.com/a/12982689
#
#   chunks = []
#   for tmp in text.split('}}'):
#       chunks.extend(re.split(r'{{c[0-9]+::', tmp))
#
#   # hacky ! on tire profit du fait que tokenizer(List[str]) re-tokenize chaque str
#   ids = tokenizer(chunks, is_split_into_words=True, max_length=512, truncation=True)
#   starts_with_cloze = int(text.startswith('{{'))
#
#   is_cloze = [0 if i is None else (i + starts_with_cloze) % 2  for i in ids.word_ids()]
#
#   ids['labels'] = is_cloze  # name of the labels is often "labels" in the fine-tuned models below
#
#   # assert len(is_cloze) == len(ids['input_ids']) (always true)
#
#   IS_CLOZE_NB += sum(is_cloze)
#   TOTAL += len(is_cloze)
#   return ids
#
# tokenized_dataset = dataset.map(process)
# print('Proportion IS_CLOZE : ', IS_CLOZE_NB/TOTAL)

"""*NB : impossible de mettre is_cloze = [-1 if i is None, ...] CAR FAIT UNE ERREUR DANS LA PERTE BCELoss.*

*TODO : régler les paramètres du tokenizer ? https://huggingface.co/docs/transformers/main_classes/tokenizer*

#### Vérif
"""

# row = dataset['train'][0]
# # text = re.sub(CLEANR, '', row['tokens'])
# text = row['field']
# chunks = []
# for tmp in text.split('}}'):
#     chunks.extend(re.split(r'{{c[0-9]+::', tmp))
# ids = tokenizer(chunks, is_split_into_words=True)
# print(text)
# print(ids)
# print(tokenizer.convert_ids_to_tokens(ids["input_ids"]))
# print(ids.word_ids())
# starts_with_cloze = int(text.startswith('{{'))
# print([0 if i is None else i % 2 + starts_with_cloze for i in ids.word_ids()])
#
# """#### Export"""
#
# tokenized_dataset['train'] = tokenized_dataset['train'].remove_columns(['field', 'file', 'usn', 'guid', 'mid', 'mod', 'id', 'tags', 'lang'])
#
# tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1)
#
# tokenized_dataset
#
# """Push dataset to Huggingface Hub"""
#
# notebook_login()  # WRITE ACCESS : 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
#
# tokenized_dataset.push_to_hub("nicolasmicaux/anki_data", private=True)

"""# Fine-tuning the model

*The warning is telling us we are throwing away some weights (the `vocab_transform` and `vocab_layer_norm` layers) and randomly initializing some other (the `pre_classifier` and `classifier` layers). This is absolutely normal in this case, because we are removing the head used to pretrain the model on a masked language modeling objective and replacing it with a new head for which we don't have pretrained weights, so the library warns us we should fine-tune this model before using it for inference, which is exactly what we are going to do.*
"""

# notebook_login()  # WRITE ACCESS : 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

from huggingface_hub import Repository, get_full_repo_name
super_model_name = model_checkpoint.split("/")[-1]
model_name = f"{super_model_name}-finetuned-anki"
output_dir = model_name
repo_name = get_full_repo_name(model_name)
repo = Repository(output_dir, clone_from=repo_name)

from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer)

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
)  # TODO : why use separate val and test ?

from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=2)

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
  'pretraining_task': task,
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
          wandb.log({'epoch': epoch, 'accuracy': results[0], 'precision': results[1], 'recall': results[2], 'f1': results[3]}, commit=False)
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
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

unwrapped_model.push_to_hub(repo_name)

tokenizer.push_to_hub(repo_name)

wandb.finish()

"""# Usage"""

from transformers import pipeline

model_checkpoint = "nicolasmicaux/xlm-roberta-base-finetuned-anki"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)  # aggregation_strategy permet une présentation plus jolie des résultats

def add_cloze(text):
  groups = token_classifier(text)
  output = ''
  for group in groups:
    if group['entity_group'] == 'LABEL_1':
      output += '{{c1::' + group['word'] + '}} '
    else:
      output += group['word'] + ' '
  return output

add_cloze("""N'importe qui peut hypnotiser, il suffit de suivre un script et de le délivrer avec le bon tempo, donc ya toute les chances que ça marche. Les formations des médecins pour induire des anesthésies (c'est pas rien quand même) ne durent que quelques jours ! 
Et d'ailleurs, ça peut être très rapide, en quelques secondes tu peux déjà être en hypnose. Tout le monde peut être hypnotisé, du moins tout ceux qui sont capables d'imaginer les choses qu'on leur propose. Et ça en fait c'est automatique dans le cerveau, ça se fait sans efforts : quand on entend des choses notre cerveau va mécaniquement les imaginer. 
D'accord pour tenter de petites suggestions ? """)

token_classifier("""Métapaquets 
Lorsqu'un package conda est utilisé uniquement pour les métadonnées et ne contient aucun fichier, il est appelé métapaquet. Le métapaquet peut contenir des dépendances vers plusieurs bibliothèques de base de bas niveau et peut contenir des liens vers des fichiers logiciels qui sont automatiquement téléchargés lors de leur exécution. Les métapaquets sont utilisés pour capturer des métadonnées et simplifier les spécifications de packages complexes.
Un exemple de métapaquet est "anaconda", qui rassemble tous les packages du programme d'installation d'Anaconda. La commande crée un environnement qui correspond exactement à ce qui serait créé à partir du programme d'installation d'Anaconda. Vous pouvez créer des métapaquets avec la commande. Incluez le nom et la version dans la commande.conda create -n envname anacondaconda metapackage""")

"""*Then we will need a data collator that will batch our processed examples together while applying padding to make them all the same size (each pad will be padded to the length of its longest example). There is a data collator for this task in the Transformers library, that not only pads the inputs, but also the labels:*

XLMRobertaForTokenClassification explained [here](https://huggingface.co/transformers/v3.5.1/model_doc/xlmroberta.html#xlmrobertafortokenclassification).

forward(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None)

A propos de la loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

# Annexes
"""

# Commented out IPython magic to ensure Python compatibility.
# Admin code (git)
# %cd /content
# !git clone https://huggingface.co/datasets/nicolasmicaux/anki_data
# # %cd anki_data
# !git lfs install
# !git rm -r data
# !git rm dataset_infos.json
# !git config --global user.email "nicolas.micaux@telecom-paris.fr"
# !git commit -m "remove data/"
# !git push

# import torch
# torch.cuda.empty_cache()
#
# from collections import Counter
# Counter([item for sublist in tokenized_dataset['test']['labels'] for item in sublist])