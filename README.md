# Halicioglu Data Science Institute Senior Capstone

## Project Overview
This DSC Capstone Project focuses on an LLM based chatbot trained on a smart home dataset. This chatbot can answer questions pretaining to the data in the dataset. Additionally, there are guardrails built on top of the chatbot that ensure a secure user experience.

## Files Overview
This repository contains the research, experimentation, and creation of our chatbot deliverable. The main deliverables can be found in the `/app` folder. More instructions on how to run the app can be found in the next section. In the `/EDA` and   `/RAG` folders, there are Jupyter Notebooks which contain the initial exploration of our smart home dataset. We utilize the `/config` files for the guardrails and the `/text2sql` folder contains a basic implementation of the Text2SQL workflow utilized in the app. `/fine-tuning` contains the previous idea for how to train our LLM on tabular data, but we switched to a Text2SQL workflow instead. The `/persistence_layer` folder contains future work that we will incorporate into our chatbot. 

## How to Install and Setup
We set up a Docker Image for users to easily pull and run the files necessary. In your terminal, run the following lines of code, and replace the strings in the brackets with your own credentials. 
1. `echo [GHCR read token] | docker login ghcr.io -u pndang --password-stdin`
2. `docker pull ghcr.io/pndang/final-project:latest`
3. docker run --rm -p 8501:8501 -e DB_USER=[DB_USER] -e DB_PASSWORD=[DB_PASSWORD] -e DB_HOST=[DB_HOST] -e DB_NAME=[DB_NAME] -e API_KEY=[API_KEY] -e DB_URI=[DB_URI] ghcr.io/pndang/final-project:latest

If everything has run successfully, then you should be greeted with the following screen in your localhost. 

Please contact eyc004@ucsd.edu, ddli@ucsd.edu, bzou@ucsd.edu, and pndang@ucsd.edu for our credentials if you need help.


## How to use the App
After running step 3 in the section above, 

- Raise flags --> YES or NO --> Text2SQL

