/*
 * ============================================================================
 *  K-means - VERSAO PARALELA EM CUDA (GPU)
 *  Trabalho de Computacao Paralela - PUC Minas
 * ============================================================================
 *
 *  Base de dados: MNIST (70.000 imagens 28x28 = 784 dimensoes), K = 10,
 *  ate 100 iteracoes - mesma entrada e mesmos parametros das outras versoes.
 *
 *  TEMPOS DE EXECUCAO
 *  -------------------------------------------------------------------------
 *   Maquina: Intel Core i7-13700K + NVIDIA GeForce RTX 4070 12 GB,
 *            WSL2, nvcc 13.3
 *   Sequencial (CPU, 1 thread)            24.986 s
 *   CUDA - so o laco do K-means            1.620 s
 *   CUDA - total (copias + transposicao)   2.007 s
 *   Speedup (laco)                        15.42
 *   Speedup (total)                       12.45
 *  -------------------------------------------------------------------------
 *
 *  Compilar:  nvcc -O2 -fmad=false kmeans_cuda.cu -o kmeans_cuda
 *  Executar:  ./kmeans_cuda mnist.tsv 10 100
 *
 *  MUDANCAS EM RELACAO A VERSAO SEQUENCIAL (procure por [CUDA] no codigo):
 *   1) Os dados sao copiados uma unica vez para a GPU antes do laco
 *      (cudaMalloc + cudaMemcpy). Dentro do laco so trafega 1 inteiro
 *      (quantos pontos mudaram), usado para testar a convergencia.
 *   2) Passo de atribuicao -> kernel "atribuir": 1 thread por ponto.
 *      - Os centroides (K*D floats) sao carregados na memoria compartilhada
 *        (__shared__) do bloco, pois todas as threads leem os mesmos valores.
 *      - Os pontos ficam TRANSPOSTOS na GPU (pontosT[j*N + i]). Assim, quando
 *        as 32 threads de um warp leem a dimensao j, elas acessam enderecos
 *        vizinhos -> acesso coalescido a memoria global.
 *      - Contador "mudaram" somado com atomicAdd.
 *   3) Passo de atualizacao -> kernel "atualizar": 1 thread por posicao
 *      (c, j) do centroide (K*D threads). Cada thread percorre os pontos e
 *      soma os que pertencem ao cluster c. Nao ha duas threads escrevendo
 *      no mesmo lugar, entao nao precisa de atomics, e a soma e feita na
 *      mesma ordem da versao sequencial (resultado identico).
 *      Aqui os pontos sao lidos no formato normal (pontos[i*D + j]): threads
 *      vizinhas tem j vizinho -> tambem coalescido.
 *   4) -fmad=false: desliga a fusao multiplica-soma (FMA) da GPU para que as
 *      distancias sejam calculadas com o mesmo arredondamento da CPU e o
 *      resultado final bata exatamente com a versao sequencial. (Opcional.)
 * ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>

/* [CUDA] macro para checar erro de toda chamada da API */
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "Erro CUDA %s (linha %d): %s\n", #x, __LINE__, \
            cudaGetErrorString(e)); exit(1); } } while (0)

#define THREADS 256 /* threads por bloco */

static double agora(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* mesma leitura da versao sequencial */
static float *ler_base(const char *nome, int *N, int *D) {
    FILE *f = fopen(nome, "r");
    if (!f) { perror(nome); exit(1); }
    size_t tam = 1 << 16;
    char *linha = (char *)malloc(tam);
    if (!fgets(linha, tam, f)) { fprintf(stderr, "arquivo vazio\n"); exit(1); }
    int colunas = 1;
    for (char *p = linha; *p; p++) if (*p == '\t') colunas++;
    *D = colunas - 1;
    int n = 0;
    while (fgets(linha, tam, f)) n++;
    *N = n;
    float *pontos = (float *)malloc((size_t)n * (*D) * sizeof(float));
    rewind(f);
    if (!fgets(linha, tam, f)) exit(1);
    for (int i = 0; i < n; i++) {
        if (!fgets(linha, tam, f)) exit(1);
        char *p = linha;
        strtol(p, &p, 10);
        for (int j = 0; j < *D; j++)
            pontos[(size_t)i * (*D) + j] = strtof(p, &p);
    }
    free(linha);
    fclose(f);
    return pontos;
}

/*
 * [CUDA] Kernel do passo 1 (atribuicao). Thread i cuida do ponto i.
 * usar_shared = 1 quando os centroides cabem na memoria compartilhada.
 */
__global__ void atribuir(const float *pontosT, const float *centroides,
                         int *rotulo, int *mudaram, int N, int D, int K,
                         int usar_shared) {
    extern __shared__ float cent_sh[];
    const float *cent = centroides;

    if (usar_shared) {
        /* todas as threads do bloco ajudam a copiar os centroides */
        for (int p = threadIdx.x; p < K * D; p += blockDim.x)
            cent_sh[p] = centroides[p];
        __syncthreads(); /* espera a copia terminar antes de usar */
        cent = cent_sh;
    }

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float melhor = FLT_MAX;
    int c_melhor = 0;
    for (int c = 0; c < K; c++) {
        float dist = 0.0f;
        for (int j = 0; j < D; j++) {
            /* pontosT[j*N + i]: threads vizinhas (i, i+1...) leem enderecos
               vizinhos -> coalescido */
            float d = pontosT[(size_t)j * N + i] - cent[c * D + j];
            dist += d * d;
        }
        if (dist < melhor) { melhor = dist; c_melhor = c; }
    }
    if (rotulo[i] != c_melhor) {
        rotulo[i] = c_melhor;
        atomicAdd(mudaram, 1); /* varias threads incrementam o mesmo contador */
    }
}

/*
 * [CUDA] Kernel do passo 2 (atualizacao). Thread p cuida da posicao
 * (c = p / D, j = p % D) dos centroides: soma a dimensao j de todos os pontos
 * do cluster c e divide pela quantidade.
 */
__global__ void atualizar(const float *pontos, const int *rotulo,
                          float *centroides, int *contagem, int N, int D, int K) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= K * D) return;
    int c = p / D, j = p % D;

    double soma = 0.0; /* double: mesma precisao da versao sequencial */
    int cont = 0;
    for (int i = 0; i < N; i++) {
        if (rotulo[i] == c) { /* rotulo[i] e o mesmo pro warp todo (broadcast) */
            soma += pontos[(size_t)i * D + j];
            cont++;
        }
    }
    if (cont > 0) centroides[p] = (float)(soma / cont); /* vazio: mantem */
    if (j == 0) contagem[c] = cont;
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

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Base: %d pontos, %d dimensoes, K = %d, GPU: %s\n", N, D, K, prop.name);

    float *centroides = (float *)malloc((size_t)K * D * sizeof(float));
    int   *rotulo     = (int *)malloc(N * sizeof(int));
    int   *contagem   = (int *)malloc(K * sizeof(int));

    /* mesma inicializacao deterministica das outras versoes */
    for (int c = 0; c < K; c++)
        memcpy(&centroides[(size_t)c * D], &pontos[(size_t)(c * (N / K)) * D],
               D * sizeof(float));
    for (int i = 0; i < N; i++) rotulo[i] = -1;

    double t_total0 = agora();

    /* [CUDA] versao transposta dos pontos para o kernel de atribuicao */
    float *pontosT = (float *)malloc((size_t)N * D * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            pontosT[(size_t)j * N + i] = pontos[(size_t)i * D + j];

    /* [CUDA] aloca e copia tudo para a GPU uma vez so */
    float *d_pontos, *d_pontosT, *d_cent;
    int *d_rotulo, *d_mudaram, *d_contagem;
    size_t bytes = (size_t)N * D * sizeof(float);
    CHECK(cudaMalloc(&d_pontos, bytes));
    CHECK(cudaMalloc(&d_pontosT, bytes));
    CHECK(cudaMalloc(&d_cent, (size_t)K * D * sizeof(float)));
    CHECK(cudaMalloc(&d_rotulo, N * sizeof(int)));
    CHECK(cudaMalloc(&d_contagem, K * sizeof(int)));
    CHECK(cudaMalloc(&d_mudaram, sizeof(int)));
    CHECK(cudaMemcpy(d_pontos, pontos, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_pontosT, pontosT, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cent, centroides, (size_t)K * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_rotulo, rotulo, N * sizeof(int), cudaMemcpyHostToDevice));

    /* [CUDA] centroides cabem na memoria compartilhada? */
    size_t shared = (size_t)K * D * sizeof(float);
    int usar_shared = shared <= prop.sharedMemPerBlock;
    if (!usar_shared) shared = 0;

    int blocos_atrib = (N + THREADS - 1) / THREADS;     /* 1 thread por ponto */
    int blocos_atual = (K * D + THREADS - 1) / THREADS; /* 1 thread por (c,j) */

    CHECK(cudaDeviceSynchronize());
    double t0 = agora();
    int iter, mudaram = 0;
    for (iter = 0; iter < max_iter; iter++) {
        CHECK(cudaMemset(d_mudaram, 0, sizeof(int)));

        /* passo 1: atribuicao */
        atribuir<<<blocos_atrib, THREADS, shared>>>(d_pontosT, d_cent, d_rotulo,
                                                    d_mudaram, N, D, K, usar_shared);
        CHECK(cudaGetLastError());

        /* passo 2: atualizacao dos centroides */
        atualizar<<<blocos_atual, THREADS>>>(d_pontos, d_rotulo, d_cent,
                                             d_contagem, N, D, K);
        CHECK(cudaGetLastError());

        /* [CUDA] traz so o contador para testar convergencia (sincroniza) */
        CHECK(cudaMemcpy(&mudaram, d_mudaram, sizeof(int), cudaMemcpyDeviceToHost));
        if (mudaram == 0) { iter++; break; }
    }
    double t1 = agora();

    /* [CUDA] traz o resultado de volta */
    CHECK(cudaMemcpy(rotulo, d_rotulo, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(centroides, d_cent, (size_t)K * D * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(contagem, d_contagem, K * sizeof(int), cudaMemcpyDeviceToHost));
    double t_total1 = agora();

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
    printf("Tempo K-means (CUDA, so o laco): %.3f s\n", t1 - t0);
    printf("Tempo total CUDA (transposicao + copias + laco): %.3f s\n", t_total1 - t_total0);

    cudaFree(d_pontos); cudaFree(d_pontosT); cudaFree(d_cent);
    cudaFree(d_rotulo); cudaFree(d_contagem); cudaFree(d_mudaram);
    free(pontos); free(pontosT); free(centroides); free(rotulo); free(contagem);
    return 0;
}
