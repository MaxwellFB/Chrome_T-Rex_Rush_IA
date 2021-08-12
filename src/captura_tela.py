"""
PRONTO
Captura frame da tela, processa imagem, salva imagem em disco e inicia thread para escutar teclado
"""

import numpy as np
import cv2
import mss.tools
import time
from threading import Thread
import os
import shutil
from captura_teclado import CapturaTeclado


class CapturaTela:
    """Inicializa thread para escutar teclado, captura tela e salva em disco"""

    def __init__(self):
        """
        :param capturar_mais: Se deseja capturar a tela inteira do jogo ou somente a parte central (sem placar e
        chao)
        """
        self.contador = 0
        self.path = '/'

        # Cria thread para escutar teclado
        self.cap_teclado = CapturaTeclado()
        self.threads = [Thread(target=self.cap_teclado.escuta_teclado)]
        for thread in self.threads:
            thread.start()

        # Captura tela do jogo rodando na maquina
        # Tela parcial do jogo
        self.monitor = {"top": 85, "left": 100, "width": 590, "height": 90}
        # Toda tela do jogo
        # self.monitor = {"top": 45, "left": 100, "width": 600, "height": 150}

    def is_salvar_model(self):
        """
        Verifica se foi solicitado para salvar model

        :return: Boolean se deve salvar model
        """
        if self.cap_teclado.salvar_model:
            self.cap_teclado.salvar_model = False
            return True
        return False

    def capturar_frame_tela(self, mostra_tela=False, colorido=False):
        """
        Pegar frame da tela, processa e retorna imagem

        :return: Frame processado e tecla manual (humano)
        """
        # Pega tecla manual, caso tenha sido pressionado (padrao = 0)
        tecla = self.cap_teclado.tecla
        # Limpa tecla caso ela tenha sido liberado (isso eh usado para nao correr risco de nao coletar alguma tecla)
        self.cap_teclado.limpar_tecla()

        # Inicia sistema para capturar frame
        with mss.mss() as sct:
            # Captura frame
            img = np.array(sct.grab(self.monitor))
            if not colorido:
                img = self._processar_imagem(img)

            if mostra_tela:
                cv2.imshow("OpenCV/Numpy normal", img)

            return img, tecla

    def preparar_salvar_frame(self, rodada, path='Frame'):
        """
        Cria pasta para armazenar frames, deve ser chamado antes de "salvar_frame"

        :param rodada: Rodada atual, sera usado como nome da pasta para armazenar os frames
        :param path: Caminho a pasta principal (as novas pastas sera criadas dentro dela)
        """
        self.path = path + '/' + str(rodada) + '/'
        # Se pasta existe, exclui
        if os.path.isdir(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
            # Demora para excluir, dependendo do tamanho da pasta, entao espera, caso contrario ira dar erro no mkdir
            time.sleep(1)
        os.mkdir(self.path)

        self.contador = 0

    def salvar_frame(self, frame, tecla, game_over):
        """
        Salva frame, nome do arquivo possui contador, tecla informada e se foi frame de game over. Método
         "preparar_salvar_frame" deve ser chamado primeiro

        :param frame: Imagem a ser salva
        :param tecla: Tecla pressionado no frame
        :param game_over: Se o frame representa game over
        """
        cv2.imwrite('{}frame{}-{}-{}.png'.format(self.path, self.contador, tecla, game_over), frame)

        self.contador += 1

    def _processar_imagem(self, image):
        """
        Processa imagem para ficar com 1 channel (grayscale) e destaca as bordas

        :return: Imagem processada
        """
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.Canny(processed_img, threshold1=250, threshold2=255)
        return processed_img

    def encerrar(self):
        cv2.destroyAllWindows()