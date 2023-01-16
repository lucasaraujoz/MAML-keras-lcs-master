import random
import numpy as np
import glob
import cv2 as cv


class MAMLDataLoader:

    def __init__(self, data_path, batch_size, n_way, k_shot, q_query):
        """
        :param data_path: caminho do dataset
        :param batch_size: numero de tarefas
        :param n_way:  qntd de classes por tarefas
        :param k_shot: qntd de imagens suporte
        :param q_query: qntd de imagens de consulta usada depois da otimização pra cada tarefa individual
        """
        print("inicio dataloader ===============================================================")
        self.file_list = [f for f in glob.glob(data_path,
                                               recursive=True)]  # ./Omniglot/images_background/**/character*
        print(self.file_list)
        print(f'TAMANHO FILE LIST = {len(self.file_list)}')        
        print(f'TAMANHO BATCH_SIZE = {batch_size}')
        self.steps = len(self.file_list) // batch_size #???
        print(f'TAMANHO STEPS = {self.steps}')
        print("=================================================================================")

        self.n_way = n_way
        print(f'self.n_way = {self.n_way}')
        #print(f'img_dirs = {random.sample(self.file_list, self.n_way)}')
        self.k_shot = k_shot
        self.q_query = q_query
        self.meta_batch_size = batch_size

    def __len__(self):
        return self.steps

    def get_one_task_data(self):
        """
        :return: support_data, query_data
        """
        # self.file_list = sao as pastas respectivas das classes
        # ex pneumonia
        # img_dirs = [Pneumonia, Covid, Hernia] se n_way = 3 monta um vetor de pastas aleatory de tamanho 3
        img_dirs = random.sample(self.file_list, self.n_way)
        print(f'img_dirs = {img_dirs}')
        support_data = [] #aux
        query_data = [] #aux

        support_image = []
        support_label = []
        query_image = []
        query_label = []
        #n importa pq aqui na amostra cada classe vai receber seu respectivo label do enumerate
        for label, img_dir in enumerate(img_dirs):  # ex: [[0, Pneumonia], [1, Covid], [2, Hernia]]
            img_list = [f for f in
                        glob.glob(img_dir + "/*.png", recursive=True)]  # img_list = [todas as imagens pra uma pasta]
            images = random.sample(img_list, self.k_shot + self.q_query)  # se kshot=query=2 images [img1,img2,img3,img4]

            # Read support set
            for img_path in images[:self.k_shot]:  #suport_image = [img1,img2]
                image = cv.imread(img_path)
                image = cv.resize(image, (128, 128))
                image = (image / 255.).astype("float32")
                #image = np.expand_dims(image, axis=-1)
                support_data.append((image, label)) #adiciona uma tupla no vetor suporte_data

            # Read query set
            for img_path in images[self.k_shot:]: #suport_image = [img3,img4]
                image = cv.imread(img_path)
                image = cv.resize(image, (128, 128))
                image = (image / 255.).astype("float32")
                #image = np.expand_dims(image, axis=-1)
                query_data.append((image, label))  #adiciona uma tupla no vetor query_data

        # shuffle support set
        random.shuffle(support_data)
        for data in support_data:
            support_image.append(data[0])  #le o suport_data q agr ta aleatorio e adiciona na support_image e support_label
            support_label.append(data[1])

        # shuffle query set
        random.shuffle(query_data)
        for data in query_data:
            query_image.append(data[0])
            query_label.append(data[1])
        print(support_label)
        return np.array(support_image), np.array(support_label), np.array(query_image), np.array(query_label)

    def get_one_batch(self):
        """
        Uma função generator que retorna um batch do tamanho especificado por meta_batch_size
        Um batch é um conjunto de tarefas diferentes de classificação
        Cada tarefa contém um subconjunto aleatório do dataset para treinamento, e outro para query.
        :return: k_shot_data, q_query_data
        """

        while True:
            batch_support_image = []
            batch_support_label = []
            batch_query_image = []
            batch_query_label = []

            for _ in range(self.meta_batch_size):  # meta_batch size = numero de tarefas
                support_image, support_label, query_image, query_label = self.get_one_task_data()  # preparado um conjunto suporte/query aleatorio do dataset fornecido
                batch_support_image.append(support_image)
                batch_support_label.append(support_label)
                batch_query_image.append(query_image)
                batch_query_label.append(query_label)
                #as tarefas sao carregadas nesse vetores, ex: batch_support_image[tarefa1]
                #quando eu for ler cada tarefa, ex: batch_support_image[0]=batch_support_label[0]=batch_query_image[0]=batch_query_label[0]
            yield np.array(batch_support_image), np.array(batch_support_label), \
                np.array(batch_query_image), np.array(batch_query_label) #yeld gerador que tbm serve com return
