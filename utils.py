from dataset import dataset
from nltk.translate.bleu_score import sentence_bleu


def itos_to_string(inp, dataset):
    strg = [dataset.vocab.itos[int(idx)] for idx in inp]
    res = " "
    res = res.join(strg)
    return res


def calc_bleu(original, predicted):
    bleuScore=0
    for idx in range(len(original)):
        bleuScore += sentence_bleu([itos_to_string(predicted[idx],dataset)], 
                              itos_to_string(original[idx],dataset), weights = [1])
    return bleuScore/len(original)

