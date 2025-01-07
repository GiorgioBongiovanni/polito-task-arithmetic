# 1) Fine-tuning su tutti i dataset
# 2) Valutazione di un singolo task
# 3) Task addition

# Attiva l'ambiente virtuale
source venv/Scripts/activate

# Imposta il PYTHONPATH per includere la directory radice
export PYTHONPATH=$(pwd)

# Esegui la valutazione per il task addition
python launch_scripts/eval_task_addition.py \
    --data-location="C:\Users\bongi\Documents\Advanced machine learning\polito-task-arithmetic\task_arithmetic_datasets" \
    --save="C:\Users\bongi\Documents\Advanced machine learning\polito-task-arithmetic\results"

# Disattiva l'ambiente virtuale
deactivate