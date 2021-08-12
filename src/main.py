from play_dino import PlayDino

if __name__ == '__main__':
    PlayDino(
        #path_model='RedeDino-Suprema.h5',
        path_model='model.h5',
        time_steps=3,
        prever=True,
        treinar=True,
        substituir_model=False,
        epsilon=0.0,
        rodada=0,
        mostra_tela=False,
        frames_por_previsao=5,
        frames_iniciais_ignorados=20,
        shape_frame=(90, 590, 1)
    )
