# NLP_Mini_Project

This is our NLP(Natural Language Processing) coruse mini-project which we call it jackfruit problem, we have worked on the project "Chatbot to respond to text queries pertaining to various Acts, Rules, and Regulations applicable to Mining industries".

First, extract.py pulls text from PDF mining laws and saves it to a MongoDB database.
Next, train.py uses this text to fine-tune a SentenceTransformer model (like all-MiniLM-L6-v2). This model learns to understand the legal language, generating training examples with positive and negative question-answer pairs to improve its ability to find relevant text.
Finally, test.py runs a Tkinter GUI. When a user asks a question, it uses the trained model to find the most relevant law passages from the database. If the relevance is too low, it dismisses the query. Otherwise, it simulates an LLM to provide a summary, risk assessment, and legal recommendations based on the retrieved texts.

### Activity Diagram

![image](https://github.com/user-attachments/assets/f290ed16-905d-4806-986a-1ffd2a2516aa)



### Architecture Diagram

![image](https://github.com/user-attachments/assets/1f15b5dd-4cbd-49ee-a975-9e9f61fcc783)

### Commands To Run


 ```shell
   python extract.py
   # or
   python3 extract.py
   # to extract information from pdf files and push it into mongoDB database
   ```

 ```shell
   python train.py
   # or
   python3 train.py
   # to train the model
   ```

### Output Screenshots

When an invalid prompt is entered
![image](https://github.com/user-attachments/assets/8c04b803-7133-45a9-a5f1-ab1e3347d70b)


When we enter a valid prompt
![image](https://github.com/user-attachments/assets/d9b9f11e-e38d-4238-bb8b-eacd12991b1e)


Training
![image](https://github.com/user-attachments/assets/0c4e4a02-001d-428a-89fd-8f17622e65ce)







Team Members:  
- Suhas Venkata Karamalaputti (PES2UG22CS590)  
- Soham Pravin Salunkhe(PES2UG22CS565)
- Mohit Prasad Singh(PES2UG22CS320)

