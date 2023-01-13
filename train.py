# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2021/4/23 17:30
# @Software: PyCharm
# @Brief: 训练脚本
import keras.optimizers
from tensorflow.keras import optimizers, utils
import tensorflow as tf
import numpy as np

from dataReader import MAMLDataLoader
from net import MAML
from config import args
import os
import optuna

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_data = MAMLDataLoader(args.train_data_dir, args.batch_size, args.n_way, args.k_shot, args.q_query)
    # val_data = MAMLDataLoader(args.val_data_dir, args.val_batch_size)
    maml = MAML(args.input_shape, args.n_way)
    # O número de verificações pode ser menor e não há necessidade de atualizar tantas vezes
    # #val_data.steps = 10
    os.system('clear')
    print("limpando")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    dataset_treinamento = train_data.get_one_batch()


    def optimizacao(trial):
        learningRateInner = trial.suggest_categorical('learning_rate_inner', [1e-5, 1e-1, 1e-3, 1e-4])
        learningRateOuter = trial.suggest_categorical('learning_rate_outer', [1e-5, 1e-1, 1e-3, 1e-4])
        inner_optimizer = keras.optimizers.Adam(learning_rate=learningRateInner)  #
        outer_optimizer = keras.optimizers.Adam(learning_rate=learningRateOuter)
        for e in range(args.epochs):
            train_progbar = utils.Progbar(1)
            print(f'Epochs {e + 1}/{args.epochs}')
            train_meta_loss, train_meta_acc = [], []
            batch_train_loss, acc = maml.train_on_batch(dataset_treinamento,
                                                        inner_optimizer,
                                                        inner_step=1,
                                                        outer_optimizer=outer_optimizer
                                                        )
            train_meta_loss.append(batch_train_loss)
            train_meta_acc.append(acc)
            train_progbar.update(1, [('loss', np.mean(train_meta_loss)), ('accuracy', np.mean(train_meta_acc))])
            ##TODO salvar modelos com perfomance > x
            ##TODO tem q ter função de teste pra retornar a perfomance
            ##TODO usar validação para tarefa especifica
            return acc  # vai ser usado no objetivo do optuna


    study = optuna.create_study(direction='maximize', study_name="otimização2", storage='sqlite:///otimização2.db',
                                load_if_exists=True)
    print("inicio optimização")
    study.optimize(optimizacao, n_trials=6)

    ###########################
    # for e in range(args.epochs):
    #
    #     train_progbar = utils.Progbar(train_data.steps)
    #     #val_progbar = utils.Progbar(val_data.steps)
    #     print('\nEpoch {}/{}'.format(e+1, args.epochs))
    #
    #     train_meta_loss = []
    #     train_meta_acc = []
    #
    #     #val_meta_loss = []
    #     #val_meta_acc = []
    #
    #     for i in range(4):  # len(self.file_list // batchsize
    #         batch_train_loss, acc = maml.train_on_batch(train_data.get_one_batch(),  # carregado todas as tarefas
    #                                                     inner_optimizer,
    #                                                     inner_step=1,
    #                                                     outer_optimizer=outer_optimizer)
    #         print(batch_train_loss, acc)
    #
    #         train_meta_loss.append(batch_train_loss)
    #         train_meta_acc.append(acc)
    #         train_progbar.update(i+1, [('loss', np.mean(train_meta_loss)),
    #                                    ('accuracy', np.mean(train_meta_acc))])
    #     print("fim treino")
    #     #acredito que o evaluate e pra testar a perfomance do maml em outros conjuntos
    #
    #     # for i in range(val_data.steps):
    #     #     batch_val_loss, val_acc = maml.train_on_batch(val_data.get_one_batch(), inner_optimizer, inner_step=3)
    #     #
    #     #     val_meta_loss.append(batch_val_loss)
    #     #     val_meta_acc.append(val_acc)
    #     #     val_progbar.update(i+1, [('val_loss', np.mean(val_meta_loss)),
    #     #                              ('val_accuracy', np.mean(val_meta_acc))])
    #
    #     maml.meta_model.save_weights("maml.h5")
