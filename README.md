# NLP_Mini_Project

This is our NLP(Natural Language Processing) coruse mini-project where we have worked on the a Chatbot that responds to text queries pertaining to various Acts, Rules, and Regulations applicable to Mining industries as well give a compliance risk analysis in case any contradicting laws exist.

First, extract.py pulls text from PDF mining laws and saves it to a MongoDB database.
Next, train.py uses this text to fine-tune a SentenceTransformer model (like all-MiniLM-L6-v2). This model learns to understand the legal language, generating training examples with positive and negative question-answer pairs to improve its ability to find relevant text.
Finally, test.py runs a Tkinter GUI. When a user asks a question, it uses the trained model to find the most relevant law passages from the database. If the relevance is too low, it dismisses the query. Otherwise, it simulates an LLM to provide a summary, risk assessment, and legal recommendations based on the retrieved texts.

### Activity Diagram

![image](https://github.com/user-attachments/assets/d6876a28-7058-4f7d-9b17-169202d7e133)


### Architecture Diagram

![image](https://github.com/user-attachments/assets/b4db0d9e-94eb-423f-a7df-1c93e341bc6d)


### Commands To Run


 ```shell
   python data_extract.py
   # or
   python3 data_extract.py
   # to extract information from pdf files and push it into mongoDB database
   ```

 ```shell
   python model_training.py
   # or
   python3 model_training.py
   # to train the model
   ```

 ```shell
   streamlit run model_testing.py
   ```

### Output Screenshots

Streamlit Portfolio


Training
![image](https://github.com/user-attachments/assets/0c4e4a02-001d-428a-89fd-8f17622e65ce)







Team Members:  
- Suhas Venkata Karamalaputti (PES2UG22CS590)  
- Soham Pravin Salunkhe(PES2UG22CS565)
- Mohit Prasad Singh(PES2UG22CS320)

