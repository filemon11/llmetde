from datasets import Dataset
from transformers import GPT2Tokenizer, DataCollatorWithPadding
from torch.utils.data.dataloader import DataLoader
import torch

import csv

from typing import Iterator, TypedDict, Generator, Callable

class EyeTrackingSentence(TypedDict):
    sentence_id : int
    words : list[str]
    nFix : list[int]
    ffd : list[float]
    gpt : list[float]
    trt : list[float]
    fixProp : list[float]

class BatchElement(TypedDict):
    sentence_id : int
    words : list[str]
    nFix : torch.Tensor
    ffd : torch.Tensor
    gpt : torch.Tensor
    trt : torch.Tensor
    fixProp : torch.Tensor
    input_ids : torch.Tensor
    word2tokenidxs : list[list[int]]

BatchList = list[BatchElement]

class Batch():
    def __init__(self, batch_list : BatchList):
        self.input_ids : torch.Tensor = torch.stack([sentence["input_ids"] for sentence in batch_list])

        self.batch_list : BatchList = batch_list

    def __getitem__(self, idx) -> BatchElement:
        return self.batch_list[idx]
    
    def __getattr__(self, name: str):
        return [sentence[name] for sentence in self.batch_list]

def generate_data(csv_file : str) -> Generator[EyeTrackingSentence, None, None]:

    data = csv.reader(open(csv_file, "r"))
    sentence = []
    nFix_sen = []
    ffd_sen = []
    gpt_sen = []
    trt_sen = []
    fixProp_sen = []
    
    index : None | int = None

    for sentence_id, word_id, word, nFix, ffd, gpt, trt, fixProp in data:
        if sentence_id == "sentence_id":
            continue
        
        sentence_id = int(sentence_id)
        nFix = float(nFix)
        ffd = float(ffd)
        gpt = float(gpt)
        trt = float(trt)
        fixProp = float(fixProp)

        if index is None:
            index = sentence_id

        if word.endswith("<EOS>"):
            word = word[:-5]

        if sentence_id > index:
            yield {"sentence_id" : sentence_id,
                   "words" : sentence,
                   "nFix" : nFix_sen,
                   "ffd" : ffd_sen,
                   "gpt" : gpt_sen,
                   "trt" : trt_sen,
                   "fixProp" : fixProp_sen}
            index += 1
            sentence = [word]
            nFix_sen = [nFix]
            ffd_sen = [ffd]
            gpt_sen = [gpt]
            trt_sen = [trt]
            fixProp_sen = [fixProp]
        
        else:
            sentence.append(word)
            nFix_sen.append(nFix)
            ffd_sen.append(ffd)
            gpt_sen.append(gpt)
            trt_sen.append(trt)
            fixProp_sen.append(fixProp)

def generate_dataset(csv_file : str) -> Dataset:
    return Dataset.from_generator(lambda : generate_data(csv_file))

# TODO: Include in dataset which token belongs to which word

def tokenise(sentence : list[str], tokeniser : GPT2Tokenizer) -> tuple[list[int], list[list[int]]]:

    def embed(sentence : list[str]) -> list[str]:
        return [tokeniser.bos_token] + sentence + [tokeniser.eos_token]

    sentence = embed(sentence)

    encoded : list[list[int]] = [tokeniser.encode(x, add_special_tokens=False) for x in sentence]

    encoded_flat : list[int] = [item for word in encoded for item in word]

    desired_output = []
    idx = 1

    for token in encoded[1:-1]:
        tokenoutput = []
        for _ in token:
            tokenoutput.append(idx)
            idx +=1
        desired_output.append(tokenoutput)

    return {"input_ids" : encoded_flat,
            "word2tokenidxs" : desired_output}

def create_dataset(csv_file : str, tokeniser : GPT2Tokenizer) -> Dataset:
    dataset : Dataset = generate_dataset(csv_file)
    
    dataset = dataset.map(lambda e : tokenise(e["words"], tokeniser))
    #dataset = dataset.remove_columns("words")

    dataset.set_format(type = 'torch')
    return dataset

def get_tokeniser() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained("gpt2", add_prefix_space=True)

def get_collator(tokeniser : GPT2Tokenizer) -> Callable[[BatchList], Batch]:

    def collator(batch : BatchList) -> Batch:
        def pad(sentence : torch.Tensor, padding_length) -> torch.Tensor:
            """Returns padded and embedded sentence and number of padding tokens used"""
            length_difference : int = padding_length - len(sentence)

            if length_difference < 0:
                sentence = sentence[:(padding_length-1)]
                sentence = torch.cat((sentence, torch.tensor([tokeniser.eos_token_id],
                                                              device = sentence.device,
                                                              dtype = sentence.dtype)))

            elif length_difference > 0:
                sentence = torch.cat((sentence, torch.tensor(length_difference * [tokeniser.eos_token_id],
                                                             device = sentence.device,
                                                             dtype = sentence.dtype)))

            return sentence
        
        max_length : int = max([len(sentence["input_ids"]) for sentence in batch])

        new_batch : Batch = []
        for sentence in batch:
            new_sentence = sentence.copy()
            new_sentence["input_ids"] = pad(new_sentence["input_ids"], max_length)
            new_batch.append(new_sentence)

        return Batch(new_batch)

    return collator

def get_dataloader(dataset : Dataset, data_collator : DataCollatorWithPadding,
                   batch_size : int = 32) -> DataLoader:
    return DataLoader(dataset, shuffle = True, batch_size = batch_size, 
                      collate_fn = data_collator)

t = get_tokeniser()
d = create_dataset("readingtimes/training_data.csv", t)
c = get_collator(t)
dl = get_dataloader(d, c, batch_size = 64)

vd = create_dataset("readingtimes/trial_data.csv", t)
vdl = get_dataloader(vd, c, batch_size = 64)