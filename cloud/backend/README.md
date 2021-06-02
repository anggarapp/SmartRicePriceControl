## Deploy to GCE (Google Compute Engine)
1. Install Virtual Environment
```
sudo pip install virtualenv virtualenvwrapper
```
2. Setup virtualenv on 'nano ~/.bashrc'
```
#virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```
3. Implement changes
```
source ~/.bashrc
```
4. creating Virtual Environment
```
mkvirtualenv dev -p python3
```
5. using virtul Environment
```
workon dev
```
6. Install Requirement
```
pip install requirement.txt
```
7. Run in Backround
```
nohup python {$your-directory}/SmartRicePriceControl/cloud/backend/main.py &
```

## Deploy on GCR (Google Cloud Run)
### Submit Docker Image
```
gcloud builds submit --tag gcr.io/{PROJECT-NAME}/{SERVICE-NAME} --project={PROJECT-NAME}
```
#### example
```
gcloud builds submit --tag gcr.io/alert-district-312603/backend-beta --project=alert-district-312603
```
### Deploy Submited Docker Image
```
gcloud run deploy --image {IMAGE-LOCATION} --platform managed --project={PROJECT-NAME}
```
#### example
```
gcloud run deploy --image gcr.io/alert-district-312603/backend-beta --platform managed --project=alert-district-312603
```