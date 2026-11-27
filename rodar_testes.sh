#!/bin/bash
# Compila e roda todas as versoes do K-means e mostra os tempos.
# Uso: ./rodar_testes.sh [arquivo] [K] [max_iter]
ARQ=${1:-mnist.tsv}
K=${2:-10}
IT=${3:-100}

gcc -O2 kmeans_seq.c -o kmeans_seq -lm || exit 1
gcc -O2 -fopenmp kmeans_omp.c -o kmeans_omp -lm || exit 1
if command -v nvcc > /dev/null; then
    nvcc -O2 -fmad=false kmeans_cuda.cu -o kmeans_cuda
fi

echo "===== Sequencial ====="
./kmeans_seq $ARQ $K $IT | tee saida_seq.txt

for T in 1 2 4 8 16 32; do
    echo "===== OpenMP $T threads ====="
    OMP_NUM_THREADS=$T ./kmeans_omp $ARQ $K $IT | tee saida_omp_$T.txt
done

if [ -x ./kmeans_cuda ]; then
    echo "===== CUDA ====="
    ./kmeans_cuda $ARQ $K $IT | tee saida_cuda.txt
fi

echo
echo "===== RESUMO ====="
SEQ=$(grep "Tempo K-means" saida_seq.txt | awk '{print $(NF-1)}')
printf "%-22s %10s %8s\n" "Versao" "Tempo(s)" "Speedup"
printf "%-22s %10s %8s\n" "Sequencial" $SEQ "1.00"
for T in 1 2 4 8 16 32; do
    V=$(grep "Tempo K-means" saida_omp_$T.txt | awk '{print $(NF-1)}')
    printf "%-22s %10s %8.2f\n" "OpenMP $T threads" $V $(echo "$SEQ / $V" | bc -l)
done
if [ -f saida_cuda.txt ] && grep -q "Tempo K-means" saida_cuda.txt; then
    V=$(grep "Tempo K-means" saida_cuda.txt | awk '{print $(NF-1)}')
    printf "%-22s %10s %8.2f\n" "CUDA (laco)" $V $(echo "$SEQ / $V" | bc -l)
fi
echo
echo "Conferencia (SSE deve ser igual em todas):"
grep -H "SSE" saida_*.txt
