import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def compute_metrics(preds, refs):
    assert len(preds) == len(refs)

    bleu1_list, bleu2_list, bleu3_list, bleu4_list = [], [], [], []
    meteor_list = []
    rouge_l_list = []

    smoother = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for pred, ref in zip(preds, refs):

        # Tokenize
        hyp = pred.split()
        ref_tokens = ref.split()
        refs_tok = [ref_tokens]

        # ==== BLEU ====
        bleu1_list.append(sentence_bleu(refs_tok, hyp, weights=(1,0,0,0), smoothing_function=smoother))
        bleu2_list.append(sentence_bleu(refs_tok, hyp, weights=(0.5,0.5,0,0), smoothing_function=smoother))
        bleu3_list.append(sentence_bleu(refs_tok, hyp, weights=(1/3,1/3,1/3,0), smoothing_function=smoother))
        bleu4_list.append(sentence_bleu(refs_tok, hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother))

        # ==== METEOR (đã sửa) ====
        meteor_list.append(
            meteor_score([ref_tokens], hyp)
        )

        # ==== ROUGE-L ====
        rouge_scores = scorer.score(ref, pred)
        rouge_l_list.append(rouge_scores["rougeL"].fmeasure)

    return {
        "BLEU@1": sum(bleu1_list) / len(bleu1_list),
        "BLEU@2": sum(bleu2_list) / len(bleu2_list),
        "BLEU@3": sum(bleu3_list) / len(bleu3_list),
        "BLEU@4": sum(bleu4_list) / len(bleu4_list),
        "ROUGE-L": sum(rouge_l_list) / len(rouge_l_list),
        "METEOR": sum(meteor_list) / len(meteor_list),
    }
