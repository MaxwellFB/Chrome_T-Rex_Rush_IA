# Chrome T-Rex Rush com Inteligência Artificial
Este projeto teve intuito de treinar uma inteligência artificial (IA) para jogar o jogo do “Dinossauro do Chrome” simulando a visão humana, sem qualquer tipo de sensores para coletar dados do ambiente e para isso foi utilizado uma recriação chama Chrome T-Rex Rush.

Busquei manter todos os arquivos importantes para o funcionamento, preservando funcionalidades para treinar uma nova rede (model) e para utilizar redes já treinadas. Tudo utilizando Keras com Tensorflow como backend, OpenCV, Numpy, mss para capturar a tela e Pynput para capturar o teclado.

## Instalações e preparações
Nesta parte irei passar todas as informações necessárias para preparar o ambiente e assim ser possível treinar uma nova rede ou utilizar a rede já treinada disponível.

### Pré-requisitos
Caso deseje utiliza Anaconda, disponibilizei meu environment, chamado “Dino” e pode ser instalar usando o comando:

    conda env create -f environmentDino.yml

####Bibliotecas:
* Python v3.8.10
  

* mss v6.1.0
* Numpy v1.19.0
* OpenCV v4.5.3
* Pynput v1.7.3
* Tensorflow v2.2.0
* Chrome_T_Rex_Rush_Like_Gym v2.0   # Mais informações abaixo

Cuidado: Devido às limitações do meu hardware utilizei Tensorflow 2.2 e identifiquei incompatibilidade com versões mais novas do Numpy (v1.21.1).

### Instalação

Pode ser instalado manualmente do Github ou rodando o seguinte comando:

    git clone https://github.com/MaxwellFB/Chrome_T_Rex_Rush_IA.git

Todo código está configurado para rodar com o jogo Chrome_T_Rex_Rush_Like_Gym, para isso é necessário baixar o repositório e deixar a pasta dentro deste projeto (src) com nome “Chrome_T_Rex_Rush_Like_Gym”, segue link para o repositório:

[Chrome_T_Rex_Rush_Like_Gym](https://github.com/MaxwellFB/Chrome_T_Rex_Rush_Like_Gym.git)

## Como rodar
Com ambiente preparado, basta executar “main.py”. Ele está configurado para criar uma rede, iniciar o jogo, começar a jogar ao pressionar espaço e aprender. Mais detalhes pode ser encontrado nas documentações dentro dos arquivos.

A qualquer momento é possível influenciar no jogo da IA usando as setas para cima (pular) e para baixo (abaixar), todas as ações feitas pelo humano serão salvas na base de dados e irão influenciar no aprendizado da IA (fica dica se quiser forçar ela a fazer algo diferente, ou assassiná-la para salvar a rede).

Para salvar o model com aprendizado atual basta apertar a tecla 's' no teclado, assim que resultar em game over será salvo a rede atual (antes do treinamento).

Para fechar todo sistema basta apertar a tecla “esc”, imediatamente tudo será interrompido e encerrado (rede não será salva).

## Edições
Irei descrever algumas alterações que podem ser realizadas para tentar suas próprias estratégias.

### Rede
Depois de realizar alguns testes, acabei adotando uma rede com destaque nas bordas usando Canny do OpenCV, LSTM e camadas densas. No arquivo “network.py” em “init” está a rede. Na estrutura atual é possível rodar em CPU sem grandes problemas.

Continuando em “network.py” no método “treinar” está o coração do treinamento, utilizei reinforcement learning. O treinamento é realizado baseado apenas na última rodada jogada, coletando alguns frames durante o jogo, quanto mais progredir no jogo maior será o reward (com limite de 20) e com frame que causou game over com reward fixo de -20. 
 
Sugiro não retirar a limitação do reward, caso contrário ira causar um overfitting gigante :D

### Frames
Todo treinamento é realizado utilizando imagens do jogo, para isso é capturado a imagem no monitor (simulando humano), então quando o jogo iniciar recomento não alterar a posição do jogo ou tampar com outras janelas, pois isso irá ser visto pelo sistema.

Existem duas etapas principais:
* Coletar dados: os frames são coletados e tratados enquanto a IA está jogando (e claro, usado para fazer as previsões), quando ela perder, todos frames capturados serão salvos em disco na pasta “Frame”, com os comandos utilizados em cada frame (nome do arquivo);
* Coletar dados armazenados em disco e treinar: será coletado os frames já tratados armazenados em disco e será utilizado para realizar o treinamento da rede.

Com isso em mente, se desejar alterar o tratamento da imagem, é possível fazer no arquivo “captura_tela” no método “_processar_imagem” (resultado pode ser visto em tempo real ativando “mostra_tela” no “main.py”). E na variável “self.monitor” (em “init”) pode ser alterado as coordenadas de captura da imagem.

CUIDADO: se alterar o tamanho da imagem, deve ser alterado no parâmetro dentro do arquivo “main.py” e dependendo a alteração pode ser necessário alterar ao carregar os frame em disco dentro de “network.py” no método “_processar_frames”.

## Meus resultado e rede treinada:
Bom, realizei diversos testes, inicialmente comecei com uma simples rede CNN, mas conforme o esperado, tive problema devido à mudança de velocidade de jogo. 

Para resolver adotei LSTM o que deixou minha rede extremamente pesada, mas melhorou os resultados (avanço \o/). Com intuito de deixar a rede mais level, apliquei Canny do OpenCV para destacar as bordas, cortei a parte inferior do chão (devia ter feito isso desde o início, hehe), retirei as camadas CNN e apliquei reinforcement learning para o treinamento. O que me resultado em uma rede leve, rodando até mesmo em CPU (Intel I5) e bons resultados (em partes).

Tentei diversas abordagens para fazer a rede aprender a usar as 3 ações disponíveis, mas os melhores resultados eram sempre quando viciava em apenas 2 ações (abaixar e pular), o que não limitou para aprender a jogar. Toda vez que começava a usar as 3 ações (ou às duas: fazer nada e pular) ela não ficava boa o suficiente para jogar eternamente.

Abaixo segue um gif da IA treinada, sendo a tela superior visão humana e inferior visão da máquina. Ela chegou em um score de mais de 42 mil, morreu porque assassinei ela :'(, com peso no coração, mas foi necessário, ela estava vicia em jogar.

![Visão Humana vs Máquina](./img/VisaoHumana-Maquina.gif)

Para rodar a rede treinada basta apontar para o arquivo “RedeDino-Suprema.h5” no arquivo “main.py” e apreciar. Recomendo desativar “treinar” e deixar o restante dos parâmetros padrões para uma melhor experiência.

## Autor
* **Maxwell F. Barbosa** - [MaxwellFB](https://github.com/MaxwellFB)

## Agradecimentos
Agradeço a você, caso tenha lido até aqui, este projeto foi criado com paixão e documentado com coração \o/