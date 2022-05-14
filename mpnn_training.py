from matplotlib import pyplot as plt
from molgym.mpnn.layers import GraphNetwork, Squeeze
from molgym.mpnn.data import make_data_loader
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks as cb
from scipy.stats import spearmanr, kendalltau
import tensorflow as tf
import numpy as np
import json

train_loader = make_data_loader('train_data.proto', shuffle_buffer=1024)
val_loader = make_data_loader('val_data.proto')
test_loader = make_data_loader('test_data.proto')

with open('atom_types.json') as fp:
    atom_type_count = len(json.load(fp))
with open('bond_types.json') as fp:
    bond_type_count = len(json.load(fp))

def build_fn(atom_features=64, message_steps=8):
    node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
    atom_types = Input(shape=(1,), name='atom', dtype='int32')
    bond_types = Input(shape=(1,), name='bond', dtype='int32')
    connectivity = Input(shape=(2,), name='connectivity', dtype='int32')
    
    # Squeeze the node graph and connectivity matrices
    snode_graph_indices = Squeeze(axis=1)(node_graph_indices)
    satom_types = Squeeze(axis=1)(atom_types)
    sbond_types = Squeeze(axis=1)(bond_types)
    
    output = GraphNetwork(atom_type_count, bond_type_count, atom_features, message_steps,
                          output_layer_sizes=[512, 256, 128],
                          atomic_contribution=False, reduce_function='max',
                          name='mpnn')([satom_types, sbond_types, snode_graph_indices, connectivity])
    
    # Scale the output
    output = Dense(1, activation='linear', name='scale')(output)
    
    return Model(inputs=[node_graph_indices, atom_types, bond_types, connectivity],
                 outputs=output)

model = build_fn(atom_features=256, message_steps=8)

ic50s = np.concatenate([x[1].numpy() for x in iter(train_loader)], axis=0)

model.get_layer('scale').set_weights([np.array([[ic50s.std()]]), np.array([ic50s.mean()])])

model.compile(Adam(InverseTimeDecay(1e-3, 64, 0.5)), 'mean_squared_error', metrics=['mean_absolute_error'])

history = model.fit(train_loader, validation_data=val_loader, epochs=200, verbose=False, 
                   shuffle=False, callbacks=[
                       cb.ModelCheckpoint('best_model.h5', save_best_only=True),
                       cb.EarlyStopping(patience=128, restore_best_weights=True),
                       cb.CSVLogger('train_log.csv'),
                       cb.TerminateOnNaN()
                   ])

model.save('./saved_models/mpnn_13_4_22')