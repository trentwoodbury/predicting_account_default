MODEL_NAME="default_predictor"
INPUT_DATA_FILE="data_storage/demo_data.json"
VERSION_NAME="v0"
gcloud ai-platform predict --model $MODEL_NAME  \
                   --version $VERSION_NAME \
                   --json-instances $INPUT_DATA_FILE
