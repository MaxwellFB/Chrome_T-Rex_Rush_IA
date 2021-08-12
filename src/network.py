"""
PRONTO
Cria/carrega model Tensorflow e Keras, realiza carregamento e tratamento nos dados e previsoes
"""

from tensorflow.keras import layers, models, optimizers
import tensorflow.keras.backend as K
import numpy as np
import cv2
import os
import ordena


class Network:
    """
    Cria/carrega model Tensorflow e Keras, realiza carregamento e tratamento nos dados e previsoes
    """

    def __init__(self, path_model=None,
                 time_steps=3,
                 substituir_model=False,
                 path_frame='Frame/',
                 shape_frame=(90, 590, 1)):
        """
        Inicia configuracoes e se model existir carrega, caso contrario cria um novo

        :param path_model: Caminho para model .h5, caso existir
        :param time_steps: Tamanho da janela que sera utilizado na LSTM (se aplica somente se for criar um model novo)
        :param substituir_model: Se deve criar um model novo, mesmo existindo model no caminho informado
        :param path_frame: Caminho para onde estao as imagens para treinamento
        :param shape_frame: Shape das imagem para treinamento (Y, X, D)
        """
        # ----- HIPERPARAMETROS PARA REINFORCEMENT LEARNING ----- #
        self.max_reward = 20
        self.reward_game_over = -20
        self.discount = 0.9
        # ------------------------------------------------------- #

        # Dimensao da imagem de entrada (no caso as imagens salvas no diretorio_frame)
        self.input_shape = (time_steps, shape_frame[0], shape_frame[1], shape_frame[2])

        # Caminho para pasta com imagens de treinamento
        self.path_frame = path_frame

        # 0 = Sem comando; 1 = Pular; 2 = Abaixar
        self.qtd_label = 3

        # Quantidade de imagens processadas sequencialmente na rede LSTM
        self.time_steps = time_steps

        if substituir_model is False and os.path.isfile(path_model):
            print('Carregando model LSTM')
            self.model = models.load_model(path_model)
            print(self.model.summary())
            print(K.eval(self.model.optimizer.get_config()))
        else:
            print('Criando model')

            self.model = models.Sequential()

            # Cria rede CNN
            # Inicialmente fiz alguns testes com CNN para nao adotei para as versoes finais. Optei por deixar a
            # estrutura, quem sabe em um futuro ele venha a ser usado novamente :D
            '''
            self.model.add(
                layers.TimeDistributed(
                    layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
                    input_shape=self.input_shape
                )
            )
            self.model.add(
                layers.TimeDistributed(
                    layers.MaxPooling2D(pool_size=2)
                )
            )
            '''

            self.model.add(
                layers.TimeDistributed(
                    layers.Flatten(),
                    input_shape=self.input_shape
                )
            )

            self.model.add(
                layers.LSTM(32, activation='relu', return_sequences=False, dropout=0.2)
            )
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.Dense(self.qtd_label))

            optimizer = optimizers.Adam(learning_rate=0.001)

            self.model.compile(loss='mse', optimizer=optimizer)

            print(self.model.summary())

    def _buscar_frames(self, rodada_treinamento):
        """
        Busca e retorna as imagens e labels dentro da pasta informada

        :param rodada_treinamento: Numero da rodada para acessar a pasta (nome da pasta)

        :return: lista frames, lista labels e lista game_over
        """
        frames = []
        labels = []
        game_over = []
        for root, dirs, files in os.walk(self.path_frame + str(rodada_treinamento)):
            files.sort(key=ordena.natural_keys)
            for file in files:
                frames.append(cv2.imread(r'{0}/{1}'.format(root, file), cv2.IMREAD_GRAYSCALE))
                # Pega comando usado no frame
                labels.append(int(file[file.find('-') + 1:-6]))
                # Pega se foi game over
                game_over.append(int(file[file.find('-') + 3:-4]))
        return frames, labels, game_over

    def _processar_frames(self, frames, labels):
        """
        Faz tratamento dos dados e retorna no formato para uma rede LSTM

        :param frames: Lista com imagens
        :param labels: Lista com os labels

        :return: Numpy array x e Numpy array y
        """
        x = np.array(frames)
        x = x.reshape(len(frames), self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        x = x / 255

        y = np.array(labels)

        return x, y

    def treinar(self, rodada_treinamento=0):
        """
        Realiza treinamento do model utilizando as imagens da pasta da rodada informada

        :param rodada_treinamento: Numero da rodada para acessar a pasta (nome da pasta)
        """
        x, y, g = self._buscar_frames(rodada_treinamento)

        inputs, targets = [], []
        for idx in np.random.choice(range(self.time_steps, len(x)-1),
                                    int((len(x)-self.time_steps)/3),
                                    replace=False):
            # Segundo elemento eh exclusivo, por isso +1
            inputs.append(x[idx+1 - self.time_steps:idx+1])
            # Pega previsao do estado atual
            targets.append(self.prever_pelo_frame(np.array(inputs[-1]))[0])

            # Se for rodada normal (nao for game over)
            if g[idx] == 0:
                # Pega maior valor da previsao do estado resultante
                Q_sa = np.max(self.prever_pelo_frame(np.array(x[idx + 1 - self.time_steps:idx + 1]))[0])
                # Quanto mais longe for, maior o reward (ate certo limite, para nao causar numeros gigantes rapidamente)
                if len(x) * 0.2 > self.max_reward:
                    targets[-1][y[idx]] = self.max_reward + self.discount * Q_sa
                else:
                    targets[-1][y[idx]] = len(x) * 0.2 + self.discount * Q_sa
            # No formato atual esta opcao nunca acontece porque game over eh somente o ultimo frame e no "choice" ele
            # esta sendo eliminado (em "len(x)-1"), pois sera coletado nas proximas linhas
            else:
                # Penalidade caso for rodada de game over
                targets[-1][y[idx]] = self.reward_game_over

        # Estruturado para a ultima imagem sempre ser o game over da rodada
        if g[-1] == 1:
            # Sempre pega o frame de game over para realizar treinamento
            inputs.append(x[len(x) - self.time_steps:])
            # Pega previsao do estado atual
            targets.append(self.prever_pelo_frame(np.array(inputs[-1]))[0])
            # Penalidade
            targets[-1][y[-1]] = self.reward_game_over

        x_train, y_train = self._processar_frames(inputs, targets)

        self.model.fit(x_train, y_train, epochs=1, batch_size=1)

    def salvar_model(self, path='model.h5'):
        """
        Salva model

        :param path: Caminho onde deseja salvar, junto com nome para o arquivo
        """
        print('Salvando model em: {}'.format(path))
        self.model.save(path)

    def prever_pelo_frame(self, imagens):
        """
        Realiza a previsao, recebendo um array de imagens no formato numpy.array

        :param imagens: Array Numpy com os frame(s) para fazer a previsao

        :return: Array Numpy com a pontucao para cada acao possivel
        """
        img = imagens.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        img = img / 255

        previsao = self.model.predict(img)

        return previsao
