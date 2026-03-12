# Arabic Poetry GPT

For this project I built a small GPT model from scratch that learns to generate Arabic poetry. The idea was to go through the full pipeline of training a language model, starting from raw text all the way to a model that can respond to instructions.

## What I did

The project has two parts. In the first part I pretrained a base model on Arabic poetry text so it learns the general structure of the language. In the second part I fine-tuned that same model on instruction-response pairs so it can actually do useful things like write a poem on a given topic or complete a verse.

Everything runs in a single Colab notebook. No external datasets needed, the data is already inside the notebook.

## How to run it

1. Open the notebook in Google Colab
2. Go to Runtime, then Change runtime type, and pick T4 GPU
3. Run all cells in order
4. Everything gets generated automatically including checkpoints, plots and sample outputs

## Data

For pretraining I used a corpus of classical and modern Arabic poetry covering themes like love, homeland, nature, wisdom and faith. It is stored as a plain UTF-8 text file.

For fine-tuning I put together 200 instruction-response pairs covering tasks like writing poems on a topic, completing a half-written verse, translating a line into English and explaining what a poem means.

## Model

I built the model from scratch without using any pretrained weights. The architecture is a decoder-only transformer with:

- Token embeddings + positional embeddings
- 4 transformer blocks
- Multi-head attention with 4 heads and causal masking so it can only look at past tokens
- Feed-forward layers with GELU activation
- Layer norm and residual connections
- The output head shares weights with the embedding layer

I used a character-level tokenizer because it handles Arabic naturally without needing a separate tokenization library.

Config I ended up using:

| setting | value |
|---------|-------|
| embed dim | 128 |
| heads | 4 |
| layers | 4 |
| ffn size | 512 |
| context length | 256 |
| vocab size | depends on data |

## Training

Pretraining ran for 30 epochs with AdamW at lr=3e-4 and cosine annealing. Fine-tuning ran for 20 epochs at lr=1e-4.

## Results

The model generates coherent Arabic poetry and can follow basic instructions after fine-tuning. Perplexity went down consistently across both phases which means the model was actually learning.

Main issues I noticed:
- At low temperature it tends to repeat phrases
- It struggles with topics that were not in the training data
- Very long generations lose coherence after a while

## Files

```
data/pretrain/data.txt              raw pretraining text
data/finetune/poetry/sft_data.json  fine-tuning pairs
checkpoints/pretrained/model.pt     saved base model
checkpoints/finetuned/model.pt      saved fine-tuned model
results/plots/                      training curves
results/sample_generations/         generated text examples
results/error_analysis.json         failure mode analysis
notebooks/arabic_gpt_poetry.ipynb   main notebook
```

##Demo video link 
https://drive.google.com/file/d/10_9QHt_xEd5l7nXtQNvUgTR6bWXAbhEV/view?usp=drive_link
