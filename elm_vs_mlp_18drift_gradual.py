from time import time
import strlearn as sl
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from elm import ExtremeLearningMachine as elm
from extremeMLP import ExtremeMLP
import numpy as np


random_states = np.random.randint(10,10000,10)
n_chunks=250

scores = np.zeros((len(random_states), 3, n_chunks)) #ELM, MLP, EELM
ttimes = np.zeros((len(random_states), 3, n_chunks))
ptimes = np.zeros((len(random_states), 3, n_chunks)) 

for r_id, r in enumerate(random_states):
    print(r_id)
        
    stream = sl.streams.StreamGenerator(n_chunks=n_chunks,
                                        chunk_size=1000,
                                        n_classes=2,
                                        n_drifts=18,
                                        n_features=20,
                                        n_informative=20,
                                        n_redundant=0,
                                        n_repeated=0,
                                        class_sep=0.5,
                                        random_state = r,
                                        concept_sigmoid_spacing=10)

    methods = [
        elm(hidden_units=100, delta=.8),
        MLPClassifier(hidden_layer_sizes=(100,)),
        ExtremeMLP(hidden_units=100, delta=.8),
        ]

    chunk_id = 0
    while chunk := stream.get_chunk():
        X, y = chunk
        print(r_id, stream.chunk_id)
        # Test
        if stream.chunk_id > 0:
            
            for m_id, m in enumerate(methods):
                a = time()
                y_pred = m.predict(X)
                ptime = time() - a
                score = accuracy_score(y, y_pred)

                scores[r_id, m_id, chunk_id]=score
                ptimes[r_id, m_id,chunk_id]=ptime

        # Train
        for m_id, m in enumerate(methods):

            a = time()
            m.partial_fit(X, y, classes=[0, 1])
            ttime = time() - a
            
            ttimes[r_id, m_id,chunk_id]=ttime
        
        chunk_id+=1

np.save('res/elm_vs_mlp_18drift_gradual.npy', scores)
