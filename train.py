from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

import torch
import numpy as np
import random

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge 
from jiwer import cer
from jiwer import cer

import collections
import math
import numpy as np
from collections import Counter

import sacrebleu

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a fixed value for the seed
seed_value = 12
set_seed(seed_value)


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('wordnet_ic')
# def calculate_corpus_bleu(references, predictions):
#     # Tokenize the sentences into words
#     references_tokenized = [[ref.split()] for ref in references]  # Each reference wrapped in another list
#     predictions_tokenized = [pred.split() for pred in predictions]
    
#     # Calculate the corpus BLEU score
#     score = corpus_bleu(references_tokenized, predictions_tokenized, smoothing_function=SmoothingFunction().method1)
    
#     return score

def n_gram_counts(text, n):
    tokens = text.split()
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def modified_precision(predicted, expected, n):
    predicted_ngrams = n_gram_counts(predicted, n)
    # Assuming there's only one expected text per predicted text, we directly pass the first element of expected.
    expected_ngrams = n_gram_counts(expected[0], n)  # Adjusted to pass a string
    
    predicted_ngrams_count = Counter(predicted_ngrams)
    expected_ngrams_count = Counter(expected_ngrams)

    match_count = 0
    for ngram in predicted_ngrams_count:
        match_count += min(predicted_ngrams_count[ngram], expected_ngrams_count.get(ngram, 0))
    
    total_count = len(predicted_ngrams)

    return match_count, total_count

def brevity_penalty(predicted, expected):
    predicted_length = len(predicted.split())
    expected_lengths = [len(e.split()) for e in expected]
    closest_length = min(expected_lengths, key=lambda ref_len: (abs(ref_len - predicted_length), ref_len))
    
    if predicted_length > closest_length:
        return 1
    else:
        return math.exp(1 - closest_length / predicted_length)

def calculate_bleu(predicted_corpus, expected_corpus, max_n=4):
    weights = [1.0 / max_n] * max_n  # Uniform weights for simplicity, can be adjusted
    p_n = [0] * max_n
    for i in range(max_n):
        sum_match_counts = 0
        sum_total_counts = 0
        for predicted, expected in zip(predicted_corpus, expected_corpus):
            match_count, total_count = modified_precision(predicted, [expected], i+1)
            sum_match_counts += match_count
            sum_total_counts += total_count

        # Smoothing: add 1 to match counts for n-grams with at least one match
        if sum_match_counts == 0:
            sum_match_counts = 1
            sum_total_counts += 1  # Avoid division by zero
        
        p_n[i] = sum_match_counts / sum_total_counts

    # Calculate brevity penalty and geometric mean of modified precisions
    bp = sum(brevity_penalty(predicted, [expected]) for predicted, expected in zip(predicted_corpus, expected_corpus)) / len(predicted_corpus)
    geo_mean = math.exp(sum(weights[i] * math.log(p_n[i]) for i in range(max_n)))

    bleu_score = bp * geo_mean
    return bleu_score

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer):
    model.eval()

    source_texts = []
    expected = []
    predicted = []


    # Indices of examples to print - first, middle, and last
    indices_to_print = [0, len(validation_ds) - 1]
    counter = 0  # Manual counter to keep track of the current index

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            if counter in indices_to_print:
                print_msg('-' * console_width)
                print_msg(f"Validation Example {counter + 1}")
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
                print_msg('-' * console_width)

            counter += 1  # Increment the manual counter

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer_wrong', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer_wrong', wer, global_step)
        writer.flush()

        # # Compute the BLEU metric
        # metric = torchmetrics.BLEUScore()
        # bleu = metric(predicted, expected)
        # writer.add_scalar('validation BLEU', bleu, global_step)
        # writer.flush()

        # bleu_custom2 = calculate_bleu(predicted, expected)
        # writer.add_scalar('validation BLEU', bleu_custom2, global_step)
        # print_msg(f"Validation BLEU: {bleu_custom2}")
        # writer.flush()

        # For BLEU Score, wrap each target sentence in a list
        expected_for_bleu = [[exp] for exp in expected]

        # blue_corprus=calculate_corpus_bleu(predicted, expected)
        # writer.add_scalar('validation BLEU_corprus', blue_corprus, global_step)
        # print_msg(f"Validation BLEU_corprus_wrong: {blue_corprus}")
        # writer.flush()

        # Calculate BLEU score
        bleu = sacrebleu.corpus_bleu(predicted, expected_for_bleu)
        print(f"BLEU score1: {bleu.score:.2f}")
        # print(f"Full report:\n{bleu}")

    # predicted_tokens = [word_tokenize(sent, language='portuguese') for sent in predicted]
    # expected_tokens = [[word_tokenize(sent, language='portuguese')] for sent in expected]  # Expected references wrapped in another list

    # # Calculate BLEU score
    # bleu_score = corpus_bleu(expected_tokens, predicted_tokens)
    # print(f"BLEU Score_1: {bleu_score}")

    tokenized_predicted = [word_tokenize(sentence, language='portuguese') for sentence in predicted]
    tokenized_expected = [word_tokenize(sentence, language='portuguese') for sentence in expected]

    # Calculate METEOR for each pair and take the average
    meteor_scores = [meteor_score([ref], pred) for ref, pred in zip(tokenized_expected, tokenized_predicted)]
    average_meteor = sum(meteor_scores) / len(meteor_scores)

    print(f"Average METEOR Score_correct: {average_meteor}")

    rouge = Rouge()
    scores = rouge.get_scores(predicted,expected, avg=True)
    print("ROUGE scores_correct:", scores)

    # cer_scores = [cer(reference, prediction) for reference, prediction in zip(expected, predicted)]
    # average_cer = sum(cer_scores) / len(cer_scores)
    # print(f"CER_correct: {average_cer}")

    # wer_scores = [wer(reference, prediction) for reference, prediction in zip(expected, predicted)]
    # average_wer = sum(wer_scores) / len(wer_scores)
    # print(f"WER_correct: {average_wer}")
    # # if predicted:
    #     print_msg(f"Data type of elements in 'predicted': {type(predicted[0])}")
    # else:
    #     print_msg("The 'predicted' list is empty.")

    # if expected:
    #     print_msg(f"Data type of elements in 'expected': {type(expected[0])}")
    # else:
    #     print_msg("The 'expected' list is empty.")

    # # Print the entire 'predicted' and 'expected' lists
    # print_msg('Predicted Outputs:')
    # print(predicted)

    # print_msg('Expected Outputs:')
    # print(expected)

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        if epoch == 49:
        # Run validation at the end of every epoch
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
