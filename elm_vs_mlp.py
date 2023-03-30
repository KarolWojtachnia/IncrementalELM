from time import time

import matplotlib.pyplot as plt
import strlearn as sl
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier

from elm import ExtremeLearningMachine as elm
from extremeMLP import ExtremeMLP
import numpy as np

if __name__ == "__main__":
    stream = sl.streams.StreamGenerator(n_chunks=1000,
                                        chunk_size=1000,
                                        n_classes=2,
                                        n_drifts=11,
                                        n_features=10,
                                        n_informative=10,
                                        n_redundant=0,
                                        n_repeated=0,
                                        random_state=123456,
                                        class_sep=0.8,
                                        concept_sigmoid_spacing=999)

    elm = elm(hidden_units=100, delta=.8)
    mlp = MLPClassifier(hidden_layer_sizes=(100,))
    ensemble = ExtremeMLP(hidden_units=100, delta=.8)

    scores_mlp = []
    scores_elm = []
    scores_ensemble = []

    ttimes_mlp = []
    ttimes_elm = []
    ttimes_ensemble = []

    ptimes_mlp = []
    ptimes_elm = []
    ptimes_ensemble = []

    mlp_reps = 1

    while chunk := stream.get_chunk():
        X, y = chunk

        # Test
        if stream.chunk_id > 0:
            # ELM
            a = time()
            y_pred = elm.predict(X)
            ptime_elm = time() - a
            score_elm = balanced_accuracy_score(y, y_pred)

            # MLP
            a = time()
            y_pred = mlp.predict(X)
            ptime_mlp = time() - a
            score_mlp = balanced_accuracy_score(y, y_pred)

            # ensemble
            a = time()
            y_pred = ensemble.predict(X)
            ptime_ensemble = time() - a
            score_ensemble = balanced_accuracy_score(y, y_pred)

            scores_mlp.append(score_mlp)
            scores_elm.append(score_elm)
            scores_ensemble.append(score_ensemble)

            print('[%04i] SCORE ELM-%.3f vs MLP-%.3f' % (
                stream.chunk_id, score_elm, score_mlp))

            ptimes_mlp.append(ptime_mlp)
            ptimes_elm.append(ptime_elm)
            ptimes_ensemble.append(ptime_ensemble)

        # Train
        # MLP
        a = time()
        for j in range(mlp_reps):
            mlp.partial_fit(X, y, classes=[0, 1])
        ttime_mlp = time() - a

        # ELM
        a = time()
        elm.partial_fit(X, y)
        ttime_elm = time() - a

        # ensemble
        a = time()
        ensemble.partial_fit(X, y)
        ttime_ensemble = time() - a

        ttimes_mlp.append(ttime_mlp)
        ttimes_elm.append(ttime_elm)
        ttimes_ensemble.append(ttime_ensemble)

    fig, ax = plt.subplots(3, 1, figsize=(14, 7))

    ax[0].plot(scores_elm, label="ELMi")
    ax[0].plot(scores_mlp, label="MLP")
    ax[0].plot(scores_ensemble, label="ensemble")
    ax[0].legend()
    ax[0].grid(ls=":")
    ax[0].set_ylim(0, 1)
    ax[0].set_title('F1-score')

    ax[1].plot(ttimes_elm, label="ELMi")
    ax[1].plot(ttimes_mlp, label="MLP")
    ax[1].plot(ttimes_ensemble, label="ensemble")
    ax[1].legend()
    ax[1].grid(ls=":")
    ax[1].set_title('training times')

    ax[2].plot(ptimes_elm, label="ELMi")
    ax[2].plot(ptimes_mlp, label="MLP")
    ax[2].plot(ptimes_ensemble, label="ensemble")
    ax[2].legend()
    ax[2].grid(ls=":")
    ax[2].set_title('prediction times')

    plt.savefig('Results/results_class_sep_08.png', format='png')
    plt.close()
    results = np.zeros((999, 3, 3))
    results[:, 0, 0] = scores_mlp
    results[:, 1, 0] = ttimes_mlp[1:]
    results[:, 2, 0] = ptimes_mlp
    results[:, 0, 1] = scores_elm
    results[:, 1, 1] = ttimes_elm[1:]
    results[:, 2, 1] = ptimes_elm
    results[:, 0, 2] = scores_ensemble
    results[:, 1, 2] = ttimes_ensemble[1:]
    results[:, 2, 2] = ptimes_ensemble
    np.save('Results/results_class_sep_08', results)

