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
    echo "batch${BATCH_SIZE}_lr${LR}_wd${WD}_balanced${BALANCED}_stop${STOPPING_CRITERION}"
}

# Funzione per eseguire la valutazione (eval_single_task.py ed eval_task_addition.py)
run_evaluation() {
    echo "Esecuzione della valutazione per l'esperimento: $EXPERIMENT_NAME"
    python launch_scripts/eval_single_task.py \
        --data-location="$DATA_LOCATION" \
        --save="$RESULTS_LOCATION" \
        --experiment-name="$EXPERIMENT_NAME" \
        --batch-size="$BATCH_SIZE" \
        --balanced="$BALANCED"

    python launch_scripts/eval_task_addition.py \
        --data-location="$DATA_LOCATION" \
        --save="$RESULTS_LOCATION" \
        --experiment-name="$EXPERIMENT_NAME" \
        --batch-size="$BATCH_SIZE" \
        --balanced="$BALANCED"
}

# Esperimenti con bilanciamento (true e false)
for BALANCED in true false; do
    # 1a) Esperimenti su batch size
    for BATCH_SIZE in 32 8 16 64 128; do
        LR=1e-4
        WD=0.0
        STOPPING_CRITERION="epochs"
        EXPERIMENT_NAME=$(generate_experiment_name)
        echo "Esperimento con batch size = $BATCH_SIZE e bilanciamento = $BALANCED"
        python launch_scripts/finetune.py \
            --data-location="$DATA_LOCATION" \
            --save="$RESULTS_LOCATION" \
            --batch-size="$BATCH_SIZE" \
            --lr="$LR" \
            --wd="$WD" \
            --balanced="$BALANCED" \
            --stopping-criterion="$STOPPING_CRITERION" \
            --experiment-name="$EXPERIMENT_NAME"
        run_evaluation
    done

    # 1b) Esperimenti su learning rate
    for LR in 5e-4 5e-5 1e-5; do
        BATCH_SIZE=32
        WD=0.0
        STOPPING_CRITERION="epochs"
        EXPERIMENT_NAME=$(generate_experiment_name)
        echo "Esperimento con learning rate = $LR e bilanciamento = $BALANCED"
        python launch_scripts/finetune.py \
            --data-location="$DATA_LOCATION" \
            --save="$RESULTS_LOCATION" \
            --batch-size="$BATCH_SIZE" \
            --lr="$LR" \
            --wd="$WD" \
            --balanced="$BALANCED" \
            --stopping-criterion="$STOPPING_CRITERION" \
            --experiment-name="$EXPERIMENT_NAME"
        run_evaluation
    done

    # 1c) Esperimenti su weight decay
    for WD in 0.001 0.01 0.1; do
        BATCH_SIZE=32
        LR=1e-4
        STOPPING_CRITERION="epochs"
        EXPERIMENT_NAME=$(generate_experiment_name)
        echo "Esperimento con weight decay = $WD e bilanciamento = $BALANCED"
        python launch_scripts/finetune.py \
            --data-location="$DATA_LOCATION" \
            --save="$RESULTS_LOCATION" \
            --batch-size="$BATCH_SIZE" \
            --lr="$LR" \
            --wd="$WD" \
            --balanced="$BALANCED" \
            --stopping-criterion="$STOPPING_CRITERION" \
            --experiment-name="$EXPERIMENT_NAME"
        run_evaluation
    done

    # 1d) Esperimenti sui criteri di arresto
    for STOPPING_CRITERION in fisher validation; do
        BATCH_SIZE=32
        LR=1e-4
        WD=0.0
        EXPERIMENT_NAME=$(generate_experiment_name)
        echo "Esperimento con stopping_criterion = $STOPPING_CRITERION e bilanciamento = $BALANCED"
        python launch_scripts/finetune.py \
            --data-location="$DATA_LOCATION" \
            --save="$RESULTS_LOCATION" \
            --batch-size="$BATCH_SIZE" \
            --lr="$LR" \
            --wd="$WD" \
            --balanced="$BALANCED" \
            --stopping-criterion="$STOPPING_CRITERION" \
            --experiment-name="$EXPERIMENT_NAME"
        run_evaluation
    done
done

# Disattiva l'ambiente virtuale
deactivate

