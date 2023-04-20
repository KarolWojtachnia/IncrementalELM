import numpy as np
import strlearn as sl
from sklearn.metrics import accuracy_score
from elm import ExtremeLearningMachine as elm

random_states = np.random.randint(10, 10000, 10)
n_chunks = 250

deltas = [0.2, 0.5, 0.8]
scores = np.zeros((len(random_states), len(deltas), n_chunks))

for r_id, r in enumerate(random_states):

    stream = sl.streams.StreamGenerator(n_chunks=n_chunks,
                                        chunk_size=1000,
                                        n_classes=2,
                                        n_drifts=18,
                                        n_features=20,
                                        n_informative=20,
                                        n_redundant=0,
                                        n_repeated=0,
                                        class_sep=0.5,
                                        y_flip=0.05,
                                        random_state=r,
                                        concept_sigmoid_spacing=10)

    methods = []
    for d in deltas:
        methods.append(elm(hidden_units=100, delta=d))

    chunk_id = 0
    while chunk := stream.get_chunk():
        X, y = chunk
        print(r_id, stream.chunk_id)
        # Test
        if stream.chunk_id > 0:
            for m_id, m in enumerate(methods):
                y_pred = m.predict(X)
                scores[r_id, m_id, chunk_id] = accuracy_score(y, y_pred)

        # Train
        for m_id, m in enumerate(methods):
            m.fit(X, y)

        chunk_id += 1

    print(r_id)
    np.save('res/update_rate_gradual.npy', scores)
