# WebWizard

## Your Own Perplexity AI Alternative with Python and LangChain
This project provides a open-source alternative to Perplexity AI using Python, Langchain and various libraries.

![web_wizard_demo](https://github.com/user-attachments/assets/875abf64-6ad1-4287-992b-058cbb59c9dd)

## Key Features

- **🔍 Web Scraping**: Utilize DuckDuckGo for internet data gathering.
- **⚡ Fast Access**: Leverage the Groq API for quick responses from various language models.
- **🗄️ Vector Storage**: Use ChromaDB for effective data storage and retrieval.
- **🖥️ User-Friendly Interface**: Built with Shiny, offering an interactive and smooth user experience.
- **✨ Embeddings**: Enhance model understanding using Google Gemini for better performance.

## Get Started:
1. **Clone the Repository:**
```bash
git clone https://github.com/MrSentinel137/WebWizard.git
cd WebWizard
```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
     or
    
    ```bash
    pip3 install -r requirements.txt
    ```

3. **Update .env File**
     Before running the program, ensure to update your .env file with your API keys:
     ```plaintext
        GROQ_API_KEY=your_groq_api_key_here
        GOOGLE_API_KEY=your_google_api_key_here
     ```

4. **Run the Program:**
    ```bash
    shiny run web_wizard.py
    ```
    or
    ```bash
    python -m shiny run web_wizard.py
    ```

## API Keys

You will need API keys to access the following services:     
- [Groq API Keys](https://console.groq.com/keys)     
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)     
