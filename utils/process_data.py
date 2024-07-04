from collections import defaultdict as ddict


def process_data(dataset,):

    sr2o_neg = ddict(set)
    for subj, obj in dataset['test_neg']:
        sr2o_neg[subj].add(obj)
    sr2o_test_neg = {k: list(v) for k, v in sr2o_neg.items()}

    sr2o = ddict(set)
    for subj, rel, obj in dataset['train']:
        sr2o[(subj, rel)].add(obj)
    sr2o_train = {k: list(v) for k, v in sr2o.items()}

    for subj, rel, obj in dataset['test']:
        sr2o[(subj, rel)].add(obj)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}


    triplets = ddict(list)

    for subj, obj in sr2o_test_neg.items():
        triplets['test_neg'].append({'triple': (subj, 0), 'label': sr2o_test_neg[subj]})

    for (subj, rel), obj in sr2o_train.items():
        triplets['train'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)]})

    for subj, rel, obj in dataset['test']:
        triplets['test'].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})

    triplets = dict(triplets)
    return triplets


def  to_triplets(triplets, entity_dict):
    l = []
    for triplet in triplets:
        s = entity_dict[triplet[0]]
        r = 0
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l


