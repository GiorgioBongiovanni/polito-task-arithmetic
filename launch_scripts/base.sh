# 1) Fine-tuning su tutti i dataset
# 2) Valutazione di un singolo task
# 3) Task addition

# load environment variables
if [ -f "variables.sh" ]; then
  source variables.sh
fi

if [[ "$ENV_MANAGER" == "conda" ]]; then
    :
else
    source venv/Scripts/activate
fi

if [ -z "$DATA_PATH" ]; then
  # If not defined, assign a value
  DATA_PATH="C:\Users\bongi\Documents\Advanced machine learning\polito-task-arithmetic\task_arithmetic_datasets"
fi

if [ -z "$RESULT_PATH" ]; then
  # If not defined, assign a value
  RESULT_PATH="./results"
fi


# Add project ROOT to PYTHONPATH. Fixing the imports would be better but this will do for the moment.
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Esegui il fine-tuning
python launch_scripts/finetune.py \
    --data-location="$DATA_PATH" \
    --save="$RESULT_PATH" \
    --batch-size=32 \
    --lr=1e-4 \
    --wd=0.0

# Esegui la valutazione su un singolo task
# python launch_scripts/eval_single_task.py \
#     --data-location="$DATA_PATH" \
#     --save="$RESULT_PATH"

# Esegui la valutazione per il task addition
# python launch_scripts/eval_task_addition.py \
#     --data-location="$DATA_PATH" \
#     --save="$RESULT_PATH"

if [[ "$ENV_MANAGER" == "conda" ]]; then
    :
else
    deactivate
fi