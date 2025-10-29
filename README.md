# 🤖 RAG Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**AI-powered document Q&A system using RAG (Retrieval Augmented Generation)**

Upload PDFs • Ask Questions • Get Accurate Answers

</div>

---

## ✨ Features

- 📄 **PDF Upload** - Process documents automatically
- 🤖 **AI Q&A** - Get accurate answers from your documents
- 📍 **Source Citations** - See which pages were used
- 🎨 **Modern UI** - Beautiful interface with animations
- 🔒 **Privacy** - Run locally with LM Studio or use OpenAI

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- [LM Studio](https://lmstudio.ai/) or OpenAI API key

### Installation

**1. Clone & Setup Backend**
```bash
git clone https://github.com/ruppppak/RAG_CHATBOT.git
cd RAG_CHATBOT/backend

python -m venv venv
venv\Scripts\activate  # Windows | source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

**2. Setup Frontend**
```bash
cd ../frontend
npm install
```

**3. Run Application**

Terminal 1:
```bash
cd backend
venv\Scripts\activate
python main.py
```

Terminal 2:
```bash
cd frontend
npm start
```

**4. Start LM Studio** (or configure OpenAI in `backend/main.py`)

**5. Open** `http://localhost:3000`

## 🛠️ Tech Stack

**Frontend:** React, Tailwind CSS  
**Backend:** FastAPI, LangChain, FAISS  
**AI:** LM Studio / OpenAI, HuggingFace Embeddings

## 📖 Usage

1. Upload a PDF document
2. Wait for processing
3. Ask questions in the chat
4. Get answers with page references

## ⚙️ Configuration

**Use OpenAI API:** Edit `backend/main.py`:
```python
os.environ["OPENAI_API_KEY"] = "your-api-key"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
```

**Adjust retrieval:** Change `k` value in `main.py`:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Kill process: `taskkill /PID <PID> /F` (Windows) |
| Module not found | Activate venv, run `pip install -r requirements.txt` |
| Upload stuck | Check backend is running at `localhost:8000` |
| Slow responses | Use smaller model or reduce `k` value |

## 📁 Project Structure

```
RAG_CHATBOT/
├── backend/
│   ├── main.py           # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── src/App.js       # React UI
│   └── package.json
└── README.md
```

## 🤝 Contributing

Contributions welcome! Fork, create a branch, and submit a PR.

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 👤 Author

**Ruppak** - [@ruppppak](https://github.com/ruppppak)

---

<div align="center">

⭐ Star this repo if you find it useful!

</div>
