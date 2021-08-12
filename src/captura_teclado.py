"""
PRONTO
Monitora teclado para realizar acoes no jogo e informar quando deve salvar o model (treinamento)
"""

from pynput.keyboard import Listener


class CapturaTeclado:
    """Monitora algumas teclas do teclado"""

    def __init__(self):
        self.salvar_model = False
        self.tecla = 0
        self.tecla_liberada = False
        self.quit = False

    def limpar_tecla(self):
        """Volta para tecla padrao"""
        if self.tecla_liberada:
            self.tecla = 0

    def escuta_teclado(self):
        """Escuta teclas"""
        with Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def _on_press(self, key):
        """Pega tecla pressionada"""
        if str(key) == 'Key.up':
            self.tecla = 1
            self.tecla_liberada = False
        elif str(key) == 'Key.down':
            self.tecla = 2
            self.tecla_liberada = False
        elif str(key) == "'s'":
            self.salvar_model = True
        elif str(key) == 'Key.esc':
            self.tecla = 3
            # Encerra Listener
            return False

    def _on_release(self, key):
        """Soltou tecla"""
        self.tecla_liberada = True
