"""
v1.0
Classe principal, controla jogo, captura de tela e solicita previsao e treinamento
"""

from Chrome_T_Rex_Rush_Like_Gym import main as dinoGame
from captura_tela import CapturaTela
from network import Network
from collections import deque
import numpy as np


class PlayDino:
    """Controla jogo, captura tela e se comunica com a rede (model)"""
    def __init__(self, path_model='model.h5',
                 time_steps=3,
                 prever=True,
                 treinar=True,
                 substituir_model=False,
                 epsilon=0.0,
                 rodada=0,
                 mostra_tela=False,
                 frames_por_previsao=5,
                 frames_iniciais_ignorados=20,
                 shape_frame=(90, 590, 1)):
        """
        Inicializa e executa sistema inteiro

        :param path_model: Caminho em disco para o modelo, caso nao existir sera criado model novo
        :param time_steps: Tamanho da janela de frames observados pela rede LSTM
        :param prever: Se sera feito previsoes pela rede, caso negativo, somente humano ira jogar
        :param treinar: Se a rede sera treinada. So sera treinada se "prever" estiver ativado
        :param substituir_model: Se deseja substituir modelo caso existir no caminho informado
        :param epsilon: Chance de realizar acao aleatoria (sera utilizado somente em rodadas pares). Entre 0.0 e 1.0
        :param rodada: Rodada do treinamento, caso seja dados sejam salvos, ira usar este valor como nome de pasta
        :param mostra_tela: Se deseja visualizar a tela do jogo capturada (frame). Se estiver jogando manual esta tela
        fica em primeiro plano, precisa clicar na tela do jogo
        :param frames_por_previsao: Quantidade de frames que serao executados a cada previsao realizada (quanto menor,
         mais frames serao coletados). Para evitar overfitting devido a muitos dados extremamente parecidos
        :param frames_iniciais_ignorados: Quantidade de frames iniciais de cada rodada que nao serao armazenados para
         serem utilizados no treinamento (nao apresentam relevancia)
        """
        self.cap_tela = CapturaTela()

        if prever:
            # 0 = nao faz nada, 1 = pula e 2 = abaixa
            acoes_game = [0, 1, 2]

            # Armazena a ultima janela para fazer previsao
            deque_steps = deque(maxlen=time_steps)

            # Inicializa rede
            rede = Network(path_model=path_model, time_steps=time_steps, substituir_model=substituir_model,
                           shape_frame=shape_frame)
            # Prepara pasta para salvar frames da rodada
            self.cap_tela.preparar_salvar_frame(rodada)

            # Local onde os frames serao armazenado antes de serem salvos em disco
            armazena_frames = []
            # Usado para ver evolucao do modelo ao longo das rodadas
            historico_armazena_frames = []

            # Apenas para declaram a variavel pois sera utilizada para iniciar o jogo
            previsao = 0

            print("Tudo pronto, pressione espaco para iniciar! ESC para fechar")

        else:
            # Apenas para declaram a variavel pois sera utilizada para iniciar o jogo
            tecla_manual = 0
            print('Mode de jogo manual, pressione espaco para comecar a jogar! ESC para fechar')

        # Inicializa game
        self.dino = dinoGame.GameDino()
        dinoGame.introscreen()
        self.dino.reset_game()

        self.isGameQuit = False
        isGameOver = False

        while not self.isGameQuit:
            #  Computador jogando
            if prever:
                # Quantos frames ira pular a cada acao realizada
                for x in range(frames_por_previsao):
                    self.isGameQuit, isGameOver = self.dino.play(previsao)
                    if self.isGameQuit or isGameOver:
                        break

                # Captura parte da tela
                frame, tecla_manual = self.cap_tela.capturar_frame_tela(mostra_tela)

                # Se for pressionado ESC no teclado o jogo eh encerrado
                if tecla_manual == 3 or self.isGameQuit:
                    self.encerrar()
                    continue

                deque_steps.append(frame)

                if isGameOver:
                    # Se for pressionado "s" no teclado, sera salvo o modelo atual em disco
                    if self.cap_tela.is_salvar_model():
                        if path_model.rfind('.h5') != 1:
                            rede.salvar_model('{}{}.h5'.format(path_model[:path_model.rfind('.h5')], rodada))
                        else:
                            rede.salvar_model('{}{}.h5'.format(path_model, rodada))

                    deque_steps = deque(maxlen=time_steps)

                    # Armazena frames e treina se atingir quantidade minima necessaria para fazer treinamento
                    # CUIDADO: Se alterar valores de coleta de frames, time steps ou frames iniciais ignorados,
                    # pode nao coletar frames suficientes quando morrer no primeiro objeto
                    if treinar and (len(armazena_frames) > frames_iniciais_ignorados + time_steps):
                        # Tira os primeiros frames que nao possuem relevancia
                        armazena_frames_temp = armazena_frames[frames_iniciais_ignorados:]

                        # Salva os frames, menos:
                        # Ultimo = ultima acao antes do game over (muito encima para conseguir alterar resultado)
                        for x in armazena_frames_temp[:-2]:
                            self.cap_tela.salvar_frame(x[0], x[1], 0)
                        # Penultimo frame representa game_over
                        self.cap_tela.salvar_frame(armazena_frames_temp[-2][0], armazena_frames_temp[-2][1], 1)

                        historico_armazena_frames.append(len(armazena_frames))

                        print('Historico jogadas:')
                        print(historico_armazena_frames)
                        print('Rodada salva na pasta: ' + str(rodada) + '. Iniciando treinamento')

                        rede.treinar(rodada)

                        rodada += 1
                        self.cap_tela.preparar_salvar_frame(rodada)

                        armazena_frames = []

                    self.dino.reset_game()
                    isGameOver = False

                # Possui o minimo de steps exigido pela rede LSTM para fazer previsoes
                elif len(deque_steps) >= time_steps:
                    # Faz previsao se nao foi pressionado tecla manualmente no teclado (humano)
                    if tecla_manual == 0:
                        # Sorteia tecla aleatoria
                        if rodada % 2 == 0 and np.random.rand() <= epsilon:
                            previsao = np.random.choice(acoes_game)
                            print("Previsao randomica: " + str(previsao))
                        # Usa conhecimento
                        else:
                            prev = rede.prever_pelo_frame(np.array(deque_steps))
                            previsao = np.argmax(prev)
                            # Se desejar mostrar pontuacao de cada acao prevista
                            # print('{} - {}'.format(prev, previsao))
                    else:
                        previsao = tecla_manual

                    armazena_frames.append([frame, previsao])
            # Humano jogando
            else:
                self.isGameQuit, isGameOver = self.dino.play(tecla_manual)
                if not isGameOver:
                    # Aqui este comando esta sendo usado apenas para pegar tecla pressionada (e mostrar a tela quando
                    # desejado)
                    _, tecla_manual = self.cap_tela.capturar_frame_tela(mostra_tela)

                # Se for pressionado ESC no teclado o jogo eh encerrado
                if tecla_manual == 3 or self.isGameQuit:
                    self.encerrar()
                    continue

    def encerrar(self):
        """Encerra todos servicos e finaliza execucao"""
        self.dino.quit()
        self.cap_tela.encerrar()
        self.isGameQuit = True
