import utils
import treetk


def attachment_scores(pred_path, gold_path, root_symbol=None):
    """
    Parameters
    ----------
    pred_path: str
    gold_path: str
    root_symbol: str, default None

    Returns
    -------
    dict[str, Any]
    """
    preds = read_arcs(pred_path)
    golds = read_arcs(gold_path)
    scores = compute_attachment_scores(preds=preds, golds=golds, root_symbol=root_symbol)
    return scores


def read_arcs(path):
    """
    Parameters
    ----------
    path: str

    Returns
    -------
    list[list[(int, int, str)]]
    """
    hyphens = utils.read_lines(path, process=lambda line: line.split()) # list[list[str]]
    arcs = [treetk.hyphens2arcs(h) for h in hyphens] # list[list[(int, int, str)]]
    return arcs


def compute_attachment_scores(preds, golds, root_symbol=None):
    """
    Parameters
    ----------
    preds: list[list[(int, int, str)]]
    golds: list[list[(int, int, str)]]
    root_symbol: str, default None

    Returns
    -------
    dict[str, Any]
    """
    assert len(preds) == len(golds)

    if root_symbol is None:
        root_symbol = "<root>"

    scores = {} # {str: Any}

    total_ok_undir = 0.0
    total_ok_unlabeled = 0.0
    total_ok_labeled = 0.0
    total_ok_root = 0.0
    total_arcs = 0.0
    for pred_arcs, gold_arcs in zip(preds, golds):
        assert len(pred_arcs) == len(gold_arcs)

        n_ok_undir = 0.0
        n_ok_unlabeled = 0.0
        n_ok_labeled = 0.0
        n_arcs = 0.0

        pred_heads = {d: (h,r) for (h,d,r) in pred_arcs}
        gold_heads = {d: (h,r) for (h,d,r) in gold_arcs}

        for d in pred_heads:
            pred_h, pred_l = pred_heads[d]
            gold_h, gold_l = gold_heads[d]
            n_arcs += 1.0
            if pred_h == gold_h or (pred_h in gold_heads and gold_heads[pred_h][0] == d):
                n_ok_undir += 1.0
            if pred_h == gold_h:
                n_ok_unlabeled += 1.0
            if pred_h == gold_h and pred_l == gold_l:
                n_ok_labeled += 1.0

        total_ok_undir += n_ok_undir
        total_ok_unlabeled += n_ok_unlabeled
        total_ok_labeled += n_ok_labeled
        total_arcs += n_arcs

        ok_root = None
        for h,d,r in gold_arcs:
            if h == 0 and r == root_symbol:
                if (h,d,r) in pred_arcs:
                    ok_root = True
                    break
                else:
                    ok_root = False
        assert not ok_root is None
        if ok_root:
            total_ok_root += 1.0

    uuas = total_ok_undir / total_arcs
    uas = total_ok_unlabeled / total_arcs
    las = total_ok_labeled / total_arcs
    ras = total_ok_root / len(preds)
    uuas_info = "%d/%d" % (total_ok_undir, total_arcs)
    uas_info = "%d/%d" % (total_ok_unlabeled, total_arcs)
    las_info = "%d/%d" % (total_ok_labeled, total_arcs)
    ras_info = "%d/%d" % (total_ok_root, len(preds))

    scores = {"LAS": las,
              "UAS": uas,
              "UUAS": uuas,
              "RA": ras,
              "LAS_info": las_info,
              "UAS_info": uas_info,
              "UUAS_info": uuas_info,
              "RA_info": ras_info}
    return scores

