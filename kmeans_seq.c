/*
 * ============================================================================
 *  K-means - VERSAO SEQUENCIAL
 *  Trabalho de Computacao Paralela - PUC Minas
 * ============================================================================
 *
 *  Base de dados: MNIST (70.000 imagens de digitos manuscritos 28x28 = 784
 *  pixels cada, valores 0..255). Cada imagem vira um ponto em R^784.
 *  Parametros usados nos testes: K = 10 clusters, ate 100 iteracoes.
 *
 *  TEMPOS DE EXECUCAO (apenas o laco do K-means, sem a leitura do arquivo)
 *  Maquina: Intel Xeon @ 2.30GHz, 2 nucleos fisicos, gcc 13.3, -O2
 *  -------------------------------------------------------------------------
 *   Versao                    Tempo (s)    Speedup
 *   Sequencial                36.960       1.00
 *   OpenMP  1 thread          40.763       0.91
 *   OpenMP  2 threads         20.115       1.84
 *   OpenMP  4 threads         20.341       1.82
 *   OpenMP  8 threads         20.419       1.81
 *   OpenMP 16 threads         20.017       1.85
 *   OpenMP 32 threads         20.442       1.81
 *   CUDA (GPU)                  ver kmeans_cuda.cu / readme.txt
 *  -------------------------------------------------------------------------
 *  OBS: esta maquina tem so 2 nucleos, entao acima de 2 threads nao ha
 *  ganho (as threads extras dividem os mesmos 2 nucleos). Para ver o
 *  speedup escalar ate 32 threads, rodar ./rodar_testes.sh no PARCODE e
 *  substituir esta tabela.
 *
 *  Compilar:  gcc -O2 kmeans_seq.c -o kmeans_seq -lm
 *  Executar:  ./kmeans_seq mnist.tsv 10 100
 *             (arquivo, K, numero maximo de iteracoes)
 * ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>

/* tempo em segundos (relogio monotono) */
static double agora(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/*
 * Le o arquivo TSV da base: 1a linha = cabecalho, 1a coluna = rotulo (digito),
 * demais colunas = pixels. Os pontos ficam num vetor so (linha por linha):
 * o ponto i, dimensao j fica em pontos[i*D + j].
 */
static float *ler_base(const char *nome, int *N, int *D) {
    FILE *f = fopen(nome, "r");
    if (!f) { perror(nome); exit(1); }

    size_t tam = 1 << 16;
    char *linha = malloc(tam);

    /* cabecalho: conta as colunas */
    if (!fgets(linha, tam, f)) { fprintf(stderr, "arquivo vazio\n"); exit(1); }
    int colunas = 1;
    for (char *p = linha; *p; p++) if (*p == '\t') colunas++;
    *D = colunas - 1; /* tira a coluna do rotulo */

    /* conta as linhas de dados */
    int n = 0;
    while (fgets(linha, tam, f)) n++;
    *N = n;

    float *pontos = malloc((size_t)n * (*D) * sizeof(float));
    rewind(f);
    if (!fgets(linha, tam, f)) exit(1); /* pula cabecalho */
    for (int i = 0; i < n; i++) {
        if (!fgets(linha, tam, f)) exit(1);
        char *p = linha;
        strtol(p, &p, 10); /* ignora o rotulo */
        for (int j = 0; j < *D; j++)
            pontos[(size_t)i * (*D) + j] = strtof(p, &p);
    }
    free(linha);
    fclose(f);
    return pontos;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("uso: %s <arquivo.tsv> <K> <max_iter>\n", argv[0]);
        return 1;
    }
    int K = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    int N, D;

    float *pontos = ler_base(argv[1], &N, &D);
    printf("Base: %d pontos, %d dimensoes, K = %d\n", N, D, K);

    float  *centroides = malloc((size_t)K * D * sizeof(float));
    double *somas      = malloc((size_t)K * D * sizeof(double));
    int    *contagem   = malloc(K * sizeof(int));
    int    *rotulo     = malloc(N * sizeof(int));

    /* Inicializacao deterministica: centroide c = ponto de indice c*N/K.
       Assim todas as versoes (seq, OpenMP, CUDA) partem do mesmo lugar e
       devem chegar no mesmo resultado. */
    for (int c = 0; c < K; c++)
        memcpy(&centroides[(size_t)c * D], &pontos[(size_t)(c * (N / K)) * D],
               D * sizeof(float));
    for (int i = 0; i < N; i++) rotulo[i] = -1;

    double t0 = agora();
    int iter, mudaram = 0;
    for (iter = 0; iter < max_iter; iter++) {

        /* ---- PASSO 1: atribuicao - cada ponto vai pro centroide mais perto ---- */
        mudaram = 0;
        for (int i = 0; i < N; i++) {
            const float *x = &pontos[(size_t)i * D];
            float melhor = FLT_MAX;
            int c_melhor = 0;
            for (int c = 0; c < K; c++) {
                const float *m = &centroides[(size_t)c * D];
                float dist = 0.0f; /* distancia euclidiana ao quadrado */
                for (int j = 0; j < D; j++) {
                    float d = x[j] - m[j];
                    dist += d * d;
                }
                if (dist < melhor) { melhor = dist; c_melhor = c; }
            }
            if (rotulo[i] != c_melhor) { rotulo[i] = c_melhor; mudaram++; }
        }

        /* ---- PASSO 2: atualizacao - centroide = media dos pontos do cluster ---- */
        memset(somas, 0, (size_t)K * D * sizeof(double));
        memset(contagem, 0, K * sizeof(int));
        for (int i = 0; i < N; i++) {
            int c = rotulo[i];
            contagem[c]++;
            for (int j = 0; j < D; j++)
                somas[(size_t)c * D + j] += pontos[(size_t)i * D + j];
        }
        for (int c = 0; c < K; c++)
            if (contagem[c] > 0) /* cluster vazio mantem o centroide antigo */
                for (int j = 0; j < D; j++)
                    centroides[(size_t)c * D + j] = somas[(size_t)c * D + j] / contagem[c];

        if (mudaram == 0) { iter++; break; } /* convergiu */
    }
    double t1 = agora();

    /* SSE (soma dos quadrados das distancias) - serve para conferir que todas
       as versoes deram o mesmo resultado */
    double sse = 0.0;
    for (int i = 0; i < N; i++) {
        const float *x = &pontos[(size_t)i * D];
        const float *m = &centroides[(size_t)rotulo[i] * D];
        for (int j = 0; j < D; j++) { double d = x[j] - m[j]; sse += d * d; }
    }

    printf("Iteracoes: %d (ultima mudou %d pontos)\n", iter, mudaram);
    printf("Tamanho dos clusters:");
    for (int c = 0; c < K; c++) printf(" %d", contagem[c]);
    printf("\nSSE final: %.6e\n", sse);
    printf("Tempo K-means (sequencial): %.3f s\n", t1 - t0);

    free(pontos); free(centroides); free(somas); free(contagem); free(rotulo);
    return 0;
}
