# Halicioglu Data Science Institute Senior Capstone

## Project Overview
This DSC Capstone Project focuses on an LLM based chatbot trained on a smart home dataset. This chatbot can answer questions pretaining to the data in the dataset. Additionally, there are guardrails built on top of the chatbot that ensure a secure user experience.

## Files Overview
This repository contains the research, experimentation, and creation of our chatbot deliverable. The main deliverables can be found in the `/app` folder. More instructions on how to run the app can be found in the next section. In the `/EDA` and   `/RAG` folders, there are Jupyter Notebooks which contain the initial exploration of our smart home dataset. We utilize the `/config` files for the guardrails and the `/text2sql` folder contains a basic implementation of the Text2SQL workflow utilized in the app. `/fine-tuning` contains the previous idea for how to train our LLM on tabular data, but we switched to a Text2SQL workflow instead. The `/persistence_layer` folder contains future work that we will incorporate into our chatbot. 

## How to Install and Setup
We set up a Docker Image for users to easily pull and run the files necessary. To interact with our chatbot, go to your terminal, run the following lines of code, and replace the strings in the brackets with your own credentials. 
1. `echo [GHCR read token] | docker login ghcr.io -u pndang --password-stdin`
2. `docker pull ghcr.io/pndang/final-project:latest`
3. `docker run --rm -p 8501:8501 -e DB_USER=[DB_USER] -e DB_PASSWORD=[DB_PASSWORD] -e DB_HOST=[DB_HOST] -e DB_NAME=[DB_NAME] -e OPENAI_API_KEY=[API_KEY] -e DB_URI=[DB_URI] ghcr.io/pndang/final-project:latest`

If everything has run successfully, then you should be greeted with the following screen in your localhost:
![Opening Screen](/images/opening_screen.png)

Please contact eyc004@ucsd.edu, ddli@ucsd.edu, bzou@ucsd.edu, and pndang@ucsd.edu for our credentials if you need help.


## How to use the App
For context, we designed this chatbot for users to be able to ask their smart home assistant questions about their energy consumption. So imagine you are the owner of a home, and this is your personal chatbot assistant which is trained on your home energy data. You can ask it questions about what parts of the house are consuming the most energy, and recommendations for energy savings.  

After running step 3 in the section above, it may take a while before you are able to load the app. This is because it loads in the models when you begin your run. As of the checkpoint, the commands will direct you towards a guardrails version of the chatbot, which will judge if your question is within the realm of our smart home dataset. You can ask questions related to energy consumption, and the agent will respond if it is within scope or not. The response may take a bit of time, so please wait. We have another chatbot prototype which interacts with Text2SQL, and we plan on chaining the two. If the user asks a question related to smart homes, we will allow it to trigger the Text2SQL sequence. Otherwise, we will reject the question. 

Here's a demo of what the conversation may look like:
![Demo Response](/images/demo.png)
