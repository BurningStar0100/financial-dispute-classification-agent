# **AI-Powered Payment Dispute Resolution Assistant**

This project is a comprehensive system designed to help fintechs and banks resolve customer payment disputes. It uses a hybrid machine learning model to classify disputes, a rule-based engine to suggest resolutions, and an LLM-powered CLI to provide natural language insights into the dispute data.

## **🚀 Project Structure**

DISPUTE\_DETECTION/  
│  
├── dataset/  
│   ├── disputes.csv          \# Raw incoming disputes from customers  
│   └── transactions.csv      \# Transaction logs for verification  
│  
├── training/  
│   └── Dispute\_Classification\_Model\_Training.ipynb \# Notebook for training the model  
│  
├── model/  
│   ├── dispute\_classifier\_model.pkl \# The saved classification model  
│   ├── tfidf\_vectorizer.pkl         \# Saved TF-IDF vectorizer  
│   ├── label\_encoder.pkl            \# Saved label encoder  
│   └── engineered\_features.pkl    \# List of engineered feature names  
│  
├── inference/  
│   ├── dispute\_classification.py \# (Task 1\) Classifies new disputes  
│   ├── resolution\_suggestion.py  \# (Task 2\) Suggests actions for classified disputes  
│   └── query\_cli.py              \# (Task 3\) The AI Assistant CLI  
│  
├── results/  
│   ├── classified\_disputes.csv \# Output of Task 1  
│   └── resolutions.csv         \# Output of Task 2  
│  
├── main.py                      \# Main script to run the entire pipeline
├── visualization.ipynb          \# chart visualization   
└── requirements.txt            \# Python dependencies

## **📋 Prerequisites**

* Python 3.8+  
* An API Key from an LLM provider (e.g., Google for Gemini).

## **🛠️ Setup**

1. **Clone the repository:**  
   git clone \<your-repo-url\>  
   cd DISPUTE\_DETECTION

2. **Install dependencies:**  
   pip install \-r requirements.txt

3. Set up your LLM API Key:  
   For the AI Assistant CLI (Task 3\) to work, you need to set your API key as an environment variable in .env file or: 
   \# On Linux/macOS  
   export OPENAI\_API\_KEY='your\_api\_key\_here'

   \# On Windows (Command Prompt)  
   set OPENAI\_API\_KEY=your\_api\_key\_here

## **⚙️ How to Run the System**

The project is divided into two main stages: a one-time training setup and the main application execution.

### **Step 1: Train the Classification Model (One-time setup)**

* Open the training/Dispute\_Classification\_Model\_Training.ipynb notebook in a Jupyter environment.  
* Execute the cells in the notebook sequentially. This will process the training data and save the four model files (.pkl) into the model/ directory.

### **Step 2: Run the Full Dispute Resolution Pipeline**

After training the model, you can run the entire inference pipeline with a single command using main.py. This script will automatically:

1. **Classify** new disputes from dataset/disputes.csv.  
2. **Suggest resolutions** for the classified disputes.  
3. **Launch the interactive AI Assistant CLI** for you to ask questions.

### **Step 3: Visualize and Analyze Results**

After running the main pipeline, you can see the Jupyter Notebook to see and create interactive visualizations of the results.

To run the pipeline:

* Make sure you have completed the training step and the model/ directory is populated.  
* Ensure your API key is set as an environment variable.  
* Execute the main script from your terminal:  
  python main.py

The script will first process the files, creating the outputs in the results/ directory, and then automatically start the interactive CLI.

**Example CLI Questions:**

* How many duplicate charges today?  
* List unresolved fraud disputes  
* Break down disputes by type  
* What is the total amount for unresolved fraud cases?

Type exit to end the CLI session.