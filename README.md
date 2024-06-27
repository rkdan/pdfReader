# Basic RAG with streamlit for Q&A on .pdf files
Set up a virtual environment, then install the requirements:
```bash
pip install -r requirements.in
```

or, if you want to install the fully-pinned versions of the requirements:
```bash
pip install -r requirements.txt
```

You'll need a huggingface API key to be available in your environment. You can set it up by running:
```bash
export HUGGINGFACEHUB_API_KEY=XXXXXXXXXXXXXXXXXXXXXX
```

# Example usage
In the terminal, run:
```
streamlit run app.py
```

Drag and drop a pdf file from your file explorer (or via the "Browse files" button). This may take a while to load. Use the file named `example.pdf`, included in this package.

You can get summaries:

![summary](https://github.com/rkdan/pdfReader/blob/main/img/summary.png?raw=True)

or ask specific questions:

![specific](https://github.com/rkdan/pdfReader/blob/main/img/specific.png?raw=True)
