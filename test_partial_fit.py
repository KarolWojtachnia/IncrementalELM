import matplotlib.pyplot as plt
import strlearn as sl
from sklearn.neural_network import MLPClassifier

from elm import ExtremeLearningMachine as elm

if __name__ == "__main__":
    stream = sl.streams.StreamGenerator(n_chunks=250,
                                        chunk_size=1000,
                                        n_classes=2,
                                        n_drifts=7,
                                        # n_features=10,
                                        # n_informative=10,
                                        # n_redundant=0,
                                        # n_repeated=0,
                                        random_state=123456,
                                        class_sep=0.5,
                                        concept_sigmoid_spacing=999)

    clfs = [
        MLPClassifier(hidden_layer_sizes=100),
        elm(weighted=False, hidden_units=100, delta=0.3),
        # extremeMLP(weighted=False, hidden_units=100, delta=0.3)

    ]
    clf_names = [
        "MLP",
        "ELM_0.3"
        # "ELM_0.05",
        # "ELM_0.08"

    ]

    # Wybrana metryka
    metrics = [sl.metrics.f1_score,
               sl.metrics.geometric_mean_score_1]

    # Nazwy metryk
    metrics_names = ["F1 score",
                     "G-mean"]

    # Inicjalizacja ewaluatora
    evaluator = sl.evaluators.TestThenTrain(metrics)

    # Uruchomienie
    evaluator.process(stream, clfs)

    # Rysowanie wykresu
    fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
    for m, metric in enumerate(metrics):
        ax[m].set_title(metrics_names[m])
        ax[m].set_ylim(0, 1)
        for i, clf in enumerate(clfs):
            ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
        plt.ylabel("Metric")
        plt.xlabel("Chunk")
        ax[m].legend()

    # ax[0].set_title("G-mean")
    # ax[0].set_ylim(0, 1)
    # for i, clf in enumerate(clfs):
    #     ax[0].plot(evaluator.scores[i, :, 0], label=clf_names[i])
    # plt.ylabel("Metric")
    # plt.xlabel("Chunk")
    # ax[0].legend()
    plt.show()

