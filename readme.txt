===============================================================================
 K-MEANS SEQUENCIAL, OpenMP E CUDA
 Trabalho de Computacao Paralela - PUC Minas
===============================================================================

1. O QUE A APLICACAO FAZ
-------------------------------------------------------------------------------
O K-means e um algoritmo de agrupamento (clustering). Ele recebe N pontos
com D dimensoes e um numero K, e divide os pontos em K grupos, de forma que
cada ponto fique no grupo cujo centro (centroide) esta mais perto dele.

O algoritmo (versao de Lloyd) e:
  0) Escolhe K pontos iniciais como centroides.
  1) ATRIBUICAO: para cada ponto, calcula a distancia euclidiana (ao
     quadrado) ate os K centroides e marca o ponto com o centroide mais
     proximo.                                   custo: O(N * K * D)
  2) ATUALIZACAO: cada centroide passa a ser a media dos pontos que foram
     marcados com ele.                          custo: O(N * D)
  3) Repete 1 e 2 ate nenhum ponto trocar de grupo ou atingir o maximo de
     iteracoes.

O passo 1 e de longe o mais caro (K vezes mais que o passo 2) e e onde esta
o maior ganho da paralelizacao.

Detalhes da nossa implementacao:
  - Inicializacao deterministica: o centroide c comeca no ponto de indice
    c*N/K. Assim as 3 versoes partem do mesmo lugar e precisam chegar no
    mesmo resultado, o que serve como teste de correcao.
  - Cluster que ficar vazio mantem o centroide anterior.
  - Distancias em float; somas dos centroides em double.
  - No fim o programa imprime: numero de iteracoes, tamanho de cada cluster,
    o SSE (soma das distancias ao quadrado de cada ponto ao seu centroide) e
    o tempo. Como o SSE e os tamanhos dos clusters sao iguais em todas as
    versoes, sabemos que a paralelizacao nao mudou o resultado.
  - O tempo medido e so o do laco do K-means (a leitura do arquivo nao entra).

2. BASE DE DADOS (REAL)
-------------------------------------------------------------------------------
MNIST: 70.000 imagens reais de digitos manuscritos (0 a 9), 28x28 pixels.
Cada imagem vira um ponto com D = 784 dimensoes (um valor 0..255 por pixel).
Usamos K = 10 (um grupo por digito, idealmente) e ate 100 iteracoes.

Formato do arquivo (TSV, separado por tab): 1a linha e o cabecalho, 1a coluna
e o rotulo do digito (ignorado pelo K-means), as 784 colunas seguintes sao
os pixels. O codigo aceita qualquer arquivo nesse formato (descobre D pelo
cabecalho e N pelo numero de linhas).

Como baixar (copia do MNIST no repositorio PMLB, ~15 MB compactado):
  wget https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/mnist/mnist.tsv.gz
  gunzip mnist.tsv.gz          # gera mnist.tsv (~128 MB)

Custo por iteracao: N*K*D = 70000 * 10 * 784 = ~549 milhoes de diferencas,
multiplicacoes e somas so no passo 1. Com 100 iteracoes a versao sequencial
leva ~37 s (requisito: no minimo 10 s).

3. ARQUIVOS
-------------------------------------------------------------------------------
  kmeans_seq.c     versao sequencial
  kmeans_omp.c     versao OpenMP (CPU multicore)
  kmeans_cuda.cu   versao CUDA (GPU)
  rodar_testes.sh  compila tudo, roda seq + OpenMP (1,2,4,8,16,32 threads)
                   + CUDA e imprime a tabela de tempos e speedups
  readme.txt       este arquivo

As mudancas feitas para paralelizar estao comentadas no codigo com as
marcas [OpenMP] e [CUDA], e resumidas no cabecalho de cada arquivo. Os
tempos tambem estao no cabecalho de cada arquivo.

4. COMO COMPILAR E EXECUTAR
-------------------------------------------------------------------------------
Parametros dos 3 programas: <arquivo.tsv> <K> <max_iter>

Sequencial:
  gcc -O2 kmeans_seq.c -o kmeans_seq -lm
  ./kmeans_seq mnist.tsv 10 100

OpenMP (escolha o numero de threads com OMP_NUM_THREADS):
  gcc -O2 -fopenmp kmeans_omp.c -o kmeans_omp -lm
  OMP_NUM_THREADS=8 ./kmeans_omp mnist.tsv 10 100

CUDA:
  nvcc -O2 -fmad=false kmeans_cuda.cu -o kmeans_cuda
  ./kmeans_cuda mnist.tsv 10 100
  (-fmad=false e opcional; ele so faz a GPU arredondar igual a CPU para o
   resultado bater exatamente com a versao sequencial)

Tudo de uma vez:
  chmod +x rodar_testes.sh
  ./rodar_testes.sh mnist.tsv 10 100

5. O QUE FOI PARALELIZADO
-------------------------------------------------------------------------------
OpenMP (kmeans_omp.c):
  - Passo 1: "#pragma omp parallel for" no laco dos pontos. Cada ponto e
    independente, entao cada thread pega um pedaco dos N pontos
    (schedule(static), ja que todo ponto custa o mesmo). O contador de
    pontos que mudaram usa reduction(+:mudaram).
  - Passo 2: se varias threads somassem direto no mesmo centroide teriamos
    condicao de corrida. Usamos reduction de vetor
    (reduction(+:somas[:K*D], contagem[:K])): cada thread soma numa copia
    privada e no fim o OpenMP junta as copias. Mais rapido que atomic ou
    critical dentro do laco.
  - Divisao soma/contagem e calculo do SSE tambem com parallel for.

CUDA (kmeans_cuda.cu):
  - Os dados vao para a GPU uma vez so, antes do laco. Dentro do laco so
    volta 1 inteiro por iteracao (quantos pontos mudaram), para testar a
    convergencia.
  - Kernel "atribuir" (passo 1): 1 thread por ponto (70.000 threads, blocos
    de 256). Os centroides (10*784 floats = 31 KB) sao copiados para a
    memoria compartilhada do bloco, ja que todas as threads leem os mesmos
    valores. Os pontos ficam transpostos na GPU (pontosT[j*N + i]) para que
    threads vizinhas leiam enderecos vizinhos (acesso coalescido).
    O contador de mudancas usa atomicAdd.
  - Kernel "atualizar" (passo 2): 1 thread por posicao (c, j) dos
    centroides (7.840 threads). Cada thread soma a dimensao j dos pontos do
    cluster c. Nenhuma posicao e escrita por duas threads, entao nao precisa
    de atomics, e a ordem da soma e a mesma da versao sequencial.

6. RESULTADOS
-------------------------------------------------------------------------------
Maquina de desenvolvimento: Intel Xeon @ 2.30GHz, 2 nucleos, 7 GB RAM,
Ubuntu 24.04, gcc 13.3 (-O2). K = 10, 100 iteracoes, MNIST completo.

  Versao               Tempo (s)   Speedup
  Sequencial            36.960      1.00
  OpenMP  1 thread      40.763      0.91
  OpenMP  2 threads     20.115      1.84
  OpenMP  4 threads     20.341      1.82
  OpenMP  8 threads     20.419      1.81
  OpenMP 16 threads     20.017      1.85
  OpenMP 32 threads     20.442      1.81
  CUDA (so o laco)      ______      ______   (GPU: ____________)

Todas as versoes deram o mesmo resultado: 100 iteracoes, SSE = 1.784651e+11,
clusters com 8281 6043 7658 6502 5477 6447 8348 9113 6266 5865 pontos.

Analise:
  - Com 2 threads o tempo cai quase pela metade (speedup 1.84), perto do
    ideal de 2. O que falta e o custo de criar/sincronizar as threads e
    juntar as copias da reduction a cada iteracao.
  - De 4 threads para cima o tempo nao melhora porque a maquina so tem 2
    nucleos: as threads extras so disputam os mesmos nucleos. Num servidor
    com mais nucleos (PARCODE) o speedup continua subindo, ja que o passo 1
    nao tem dependencia entre pontos. A tabela deve ser refeita la com
    ./rodar_testes.sh.
  - A versao CUDA compila (testado com nvcc 12.0), mas a maquina de
    desenvolvimento nao tem GPU NVIDIA; o tempo da GPU deve ser medido no
    PARCODE e anotado aqui e no cabecalho de kmeans_cuda.cu.
