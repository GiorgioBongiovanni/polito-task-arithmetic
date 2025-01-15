#!/bin/bash

# Attiva l'ambiente virtuale
source venv/Scripts/activate

# Imposta il PYTHONPATH per includere la directory radice
export PYTHONPATH=$(pwd)

# Percorsi comuni (racchiusi tra virgolette per gestire eventuali spazi)
DATA_LOCATION="C:\Users\bongi\Documents\Advanced machine learning\polito-task-arithmetic\task_arithmetic_datasets"
RESULTS_LOCATION="C:\Users\bongi\Documents\Advanced machine learning\polito-task-arithmetic\results"

# Funzione per generare un nome univoco per l'esperimento
generate_experiment_name() {
    echo "batch${BATCH_SIZE}_lr${LR}_wd${WD}"
}

# Funzione per eseguire la valutazione (eval_single_task.py ed eval_task_addition.py)
run_evaluation() {
    echo "Esecuzione della valutazione per l'esperimento: $EXPERIMENT_NAME"
    python launch_scripts/eval_single_task.py \
        --data-location="$DATA_LOCATION" \
        --save="$RESULTS_LOCATION" \
        --experiment-name="$EXPERIMENT_NAME"

    python launch_scripts/eval_task_addition.py \
        --data-location="$DATA_LOCATION" \
        --save="$RESULTS_LOCATION" \
        --experiment-name="$EXPERIMENT_NAME"
}

# 4) Esperimenti sui 2 nuovi stopping criteria
for STOPPING_CRITERION in fisher validation; do
    BATCH_SIZE=32
    LR=1e-4
    WD=0.0
    EXPERIMENT_NAME=$(generate_experiment_name)
    echo "Esperimento con stopping_criterion = $STOPPING_CRITERION"
    python launch_scripts/finetune.py \
        --data-location="$DATA_LOCATION" \
        --save="$RESULTS_LOCATION" \
        --batch-size="$BATCH_SIZE" \
        --lr="$LR" \
        --wd="$WD" \
        --stopping-criterion="$STOPPING_CRITERION" \
        --experiment-name="$EXPERIMENT_NAME"
    run_evaluation
done

# Disattiva l'ambiente virtuale
deactivate