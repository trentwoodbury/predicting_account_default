# first I add my custom prediction pipeline to gcp
gsutil cp dist/predict_default_probability-0.1.tar.gz gs://default_model_storage_buckeet

# Then I create a new model
gcloud ai-platform models create default_predictor \
  --regions europe-west1

# Make sure we have beta
gcloud components install beta

# create a version for the model
gcloud beta ai-platform versions create v0 \
  --model default_predictor \
  --runtime-version 1.14 \
  --python-version 3.5 \
  --origin gs://default_model_storage_buckeet \
  --package-uris gs://default_model_storage_buckeet/predict_default_probability-0.1.tar.gz	 \
  --prediction-class predictor.MyPredictor

# Set environment variables
MODEL_NAME="default_predictor"
INPUT_DATA_FILE="/Users/TrentWoodbury/Code/predicting_account_default/gcp_api/data_storage/demo_data.json"
VERSION_NAME="v0"

# get some test predictions
gcloud ai-platform predict --model $MODEL_NAME  \
                   --version $VERSION_NAME \
                   --json-instances $INPUT_DATA_FILE
