"""
Avaliasaun klaster K-Means ho kategoria referénsia: akurasia, presizaun, recall no F1-score.

K-Means la iha label. Atu hetan métrika klasifikasaun:
  1. Dadus fahe ba treinu no teste.
  2. Kada klaster simu kategoria ne'ebé dominante iha dadus treinu (votu maioria).
  3. Notísia teste ida-idak prediz ho kategoria husi klaster nian; kompara ho kategoria referénsia.
Kategoria referénsia mai husi klasifikadór liafuan-xave (`category_classifier`), la'ós label umanu;
entaun métrika sukat konkordánsia ho referénsia ne'e, la'ós lorloos absolutu.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def majority_mapping(clusters, labels) -> dict:
    """{klaster: kategoria dominante}. Klaster mamuk uza kategoria dominante globál."""
    clusters, labels = np.asarray(clusters), np.asarray(labels)
    overall = _mode(labels)
    mapping = {}
    for c in np.unique(clusters):
        mapping[int(c)] = _mode(labels[clusters == c]) if (clusters == c).any() else overall
    return mapping


def _mode(values):
    vals, counts = np.unique(np.asarray(values), return_counts=True)
    return str(vals[np.argmax(counts)])


def evaluate_clusters(train_clusters, train_labels, test_clusters, test_labels, *,
                      exclude_labels=(), n_clusters=None) -> dict:
    train_clusters, test_clusters = np.asarray(train_clusters), np.asarray(test_clusters)
    train_labels, test_labels = np.asarray(train_labels).astype(str), np.asarray(test_labels).astype(str)
    if exclude_labels:
        keep_tr = ~np.isin(train_labels, list(exclude_labels))
        keep_te = ~np.isin(test_labels, list(exclude_labels))
        train_clusters, train_labels = train_clusters[keep_tr], train_labels[keep_tr]
        test_clusters, test_labels = test_clusters[keep_te], test_labels[keep_te]
    if len(train_labels) == 0 or len(test_labels) == 0:
        raise ValueError("Dadus la to'o atu avalia (treinu ka teste mamuk depois hasai kategoria).")

    mapping = majority_mapping(train_clusters, train_labels)
    default = _mode(train_labels)
    predicted = np.array([mapping.get(int(c), default) for c in test_clusters])

    classes = sorted(set(train_labels) | set(test_labels) | set(predicted))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(test_labels, predicted, labels=classes,
                                                       average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(test_labels, predicted, labels=classes,
                                                       average="weighted", zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(test_labels, predicted, labels=classes,
                                                         average=None, zero_division=0)
    baseline = float(np.mean(test_labels == default))
    cm = confusion_matrix(test_labels, predicted, labels=classes)
    return {
        "accuracy": float(accuracy_score(test_labels, predicted)),
        "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "baseline_accuracy": baseline, "baseline_label": default,
        "classes": classes,
        "per_class": [{"kategoria": c, "presizaun": float(p), "recall": float(r), "f1": float(f), "suporta": int(s)}
                      for c, p, r, f, s in zip(classes, p_c, r_c, f_c, s_c)],
        "confusion": cm.tolist(),
        "mapping": {str(k): v for k, v in mapping.items()},
        "n_train": int(len(train_labels)), "n_test": int(len(test_labels)),
        "excluded": list(exclude_labels),
    }
