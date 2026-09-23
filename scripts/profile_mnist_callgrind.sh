#!/usr/bin/env sh
set -eu

# Profila l'eseguibile di benchmark MNIST con Valgrind Callgrind.
#
# Uso (dalla root del progetto):
#   BUILD_DIR=build-wsl-openblas ./scripts/profile_mnist_callgrind.sh [percorso/eseguibile]
#
# Variabili d'ambiente opzionali:
#   BUILD_DIR=build-wsl-openblas   cartella di build (relativa alla root o assoluta)
#   CALLGRIND_OUT_DIR=callgrind-results
#   CALLGRIND_THRESHOLD=99         % cumulativo di costo mostrato da callgrind_annotate
#   BENCH_MAX_BATCHES=2            quanti batch far girare (letto dal C++, se l'hai patchato)
#   ANNOTATE=0                     metti 1 per generare anche il report riga-per-riga (enorme)
#
# Serve una build con simboli di debug (RelWithDebInfo o -g), altrimenti
# Callgrind vede solo indirizzi grezzi.

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"build-wsl-openblas"}
case "$BUILD_DIR" in
    /*) ;;
    *) BUILD_DIR="$PROJECT_ROOT/$BUILD_DIR" ;;
esac
OUTPUT_DIR=${CALLGRIND_OUT_DIR:-"$PROJECT_ROOT/callgrind-results"}
THRESHOLD=${CALLGRIND_THRESHOLD:-99}
ANNOTATE=${ANNOTATE:-0}

# Sotto Valgrind i thread vengono serializzati e quelli di OpenBLAS fanno
# spin-wait: con un solo thread il profilo e' molto piu' veloce e leggibile.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Valgrind non supporta le istruzioni AVX512: forziamo un kernel AVX2.
export OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-Haswell}
# Pochi batch bastano: ogni batch fa gli stessi conti.
export BENCH_MAX_BATCHES=${BENCH_MAX_BATCHES:-2}

if [ $# -ge 1 ]; then
    EXECUTABLE=$1
else
    EXECUTABLE="$BUILD_DIR/benchmarc_mnist"
    if [ ! -x "$EXECUTABLE" ]; then
        EXECUTABLE=$(find "$BUILD_DIR" -type f -name benchmarc_mnist -perm -u+x 2>/dev/null | head -n 1 || true)
    fi
fi

if ! command -v valgrind >/dev/null 2>&1; then
    echo "Errore: valgrind non e' installato (sudo apt install valgrind)." >&2
    exit 1
fi

if ! command -v callgrind_annotate >/dev/null 2>&1; then
    echo "Errore: callgrind_annotate non e' nel PATH (arriva con valgrind)." >&2
    exit 1
fi

if [ -z "${EXECUTABLE:-}" ] || [ ! -x "$EXECUTABLE" ]; then
    echo "Errore: eseguibile non trovato in $BUILD_DIR" >&2
    echo "Compila prima, oppure passa il percorso come primo argomento." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
CALLGRIND_FILE="$OUTPUT_DIR/callgrind.out"
SELF_REPORT="$OUTPUT_DIR/report-self.txt"
INCLUSIVE_REPORT="$OUTPUT_DIR/report-inclusive.txt"
ANNOTATED_REPORT="$OUTPUT_DIR/report-annotated.txt"

echo "Eseguibile:        $EXECUTABLE"
echo "Dati Callgrind:    $CALLGRIND_FILE"
echo "Batch per fase:    $BENCH_MAX_BATCHES (funziona solo se il C++ legge BENCH_MAX_BATCHES)"
echo "Nota: Callgrind conta istruzioni, non tempo reale. Aspettati qualche minuto."
echo

valgrind \
    --tool=callgrind \
    --callgrind-out-file="$CALLGRIND_FILE" \
    --dump-line=yes \
    "$EXECUTABLE"

echo
echo "=== Costo self: peso di ogni funzione da sola, senza le chiamate ==="
callgrind_annotate \
    --threshold="$THRESHOLD" \
    "$CALLGRIND_FILE" | tee "$SELF_REPORT"

echo
echo "=== Costo inclusivo: funzione + tutto cio' che chiama ==="
callgrind_annotate \
    --inclusive=yes \
    --threshold="$THRESHOLD" \
    "$CALLGRIND_FILE" | tee "$INCLUSIVE_REPORT"

if [ "$ANNOTATE" = "1" ]; then
    echo
    echo "=== Annotazione riga per riga (serve -g nella build) ==="
    callgrind_annotate \
        --auto=yes \
        --threshold="$THRESHOLD" \
        "$CALLGRIND_FILE" > "$ANNOTATED_REPORT"
    echo "Salvato in $ANNOTATED_REPORT"
fi

echo
echo "Fatto. Report salvati in $OUTPUT_DIR:"
echo "  report-self.txt       (costo self per funzione)"
echo "  report-inclusive.txt  (costo inclusivo per funzione)"