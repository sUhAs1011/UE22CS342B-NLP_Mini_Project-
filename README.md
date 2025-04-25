# NLP_Mini_Project

This is our NLP(Natural Language Processing) coruse mini-project which we call it jackfruit problem, we have worked on the project "Chatbot to respond to text queries pertaining to various Acts, Rules, and Regulations applicable to Mining industries".

In our project our NLP model analyze mining laws, regulations, and compliance rules. It processes user queries, retrieves relevant legal provisions, and detects contradictions between different laws. In case any conflict exists, the bot flags it and suggests resolutions accordingly. Additionally, it also provides risk assessments and legal recommendations to ensure mining companies remain compliant with regulatory frameworks.

Our model used sentence transformer and cosine similarity as the primary methodologies to make the project and the sentence transformer model fetches the similar documents based on cosine similarity.

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
![image](https://github.com/user-attachments/assets/e935d900-05d2-4616-a30e-1e542e30eb1e)


When we enter a valid prompt
![image](https://github.com/user-attachments/assets/7ae55a44-ccb6-4d82-80c0-2ffc51af87b0)






Team Members:  
- Suhas Venkata Karamalaputti (PES2UG22CS590)  
- Soham Pravin Salunkhe(PES2UG22CS565)
- Mohit Prasad Singh(PES2UG22CS320)

