from rouge_score import rouge_scorer

def compute_metrics(preds, refs):
    assert len(preds) == len(refs)

    rouge_l_list = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        rouge_l_list.append(scores["rougeL"].fmeasure)

    return {
        "ROUGE-L": sum(rouge_l_list) / len(rouge_l_list)
    }
