import numpy as np
import strlearn as sl
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from elm import ExtremeLearningMachine as elm

if __name__ == "__main__":

    stream = sl.streams.StreamGenerator(n_chunks=1000,
                                        chunk_size=1000,
                                        n_classes=2,
                                        n_drifts=11,
                                        n_features=20,
                                        n_informative=20,
                                        n_redundant=0,
                                        n_repeated=0,
                                        random_state=123456,
                                        class_sep=0.5,
                                        concept_sigmoid_spacing=999)

    elm_03 = elm(hidden_units=100, delta=.3)
    elm_05 = elm(hidden_units=100, delta=.5)
    elm_08 = elm(hidden_units=100, delta=.8)

    scores_03 = []
    scores_05 = []
    scores_08 = []

    mlp_reps = 1

    while chunk := stream.get_chunk():
        X, y = chunk

        # Test
        if stream.chunk_id > 0:
            # 03
            y_pred = elm_03.predict(X)
            score_03 = f1_score(y, y_pred)

            # 05
            y_pred = elm_05.predict(X)
            score_05 = f1_score(y, y_pred)

            # 08
            y_pred = elm_08.predict(X)
            score_08 = f1_score(y, y_pred)

            scores_03.append(score_03)
            scores_05.append(score_05)
            scores_08.append(score_08)

            print('[%04i] SCORE 03-%.3f vs 05-%.3f vs 08-%.3f' % (
                stream.chunk_id, score_03, score_05, score_08))

        # Train
        elm_03.partial_fit(X, y)
        elm_05.partial_fit(X, y)
        elm_08.partial_fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    ax.plot(scores_03, label="update rate 0.3")
    ax.plot(scores_05, label="update rate 0.5")
    ax.plot(scores_08, label="update rate 0.8")
    ax.legend()
    ax.grid(ls=":")
    ax.set_ylim(0, 1)
    ax.set_title('F1-score for different update rate')
    ax.set_xlabel('Chunk number')
    ax.set_ylabel('F1-score')

    plt.savefig('Results/results_update_rate.png', format='png')
    plt.close()

    results_array = np.array([scores_03, scores_05, scores_08])
    np.save('Results/results_update_rate', results_array)
