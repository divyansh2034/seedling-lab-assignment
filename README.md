# 🌱 Seedling Labs – AI Powered GitHub Issue Assistant

This project is an AI-powered tool that analyzes GitHub issues and produces a
structured summary, classification, priority score, labels, and potential impact.
It helps engineering teams triage faster and maintain development velocity.

---

## ✨ Features

- 🤖 Automatic issue analysis using HuggingFace LLM (Free)
- 🧠 Extracts:
  - Summary of the issue
  - Issue type classification
  - Priority score + justification
  - Suggested labels
  - Potential impact
- 📊 Shows repository metadata:
  - Owner
  - Stars
  - Open issues
- 🧾 Download structured JSON output
- 📚 History of analyzed issues
- ♻️ One-click re-run of old issues
- 🧱 Professional UX and error handling:
  - Invalid repo and issue detection
  - Backend health indicator

---

## 🧱 Architecture

```

Streamlit (Frontend)
↓
FastAPI Backend
↓
GitHub REST API
↓
HuggingFace LLM
↓
Pydantic JSON Validation
↓
Structured JSON Output + UI

```

---

## 📁 Folder Structure

```

seedling-lab-assignment/
│
├── backend/
│   ├── main.py
│
├── frontend/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md

```

---

## 🧠 Prompt Engineering Strategy

- Prompt enforces consistent and strictly formatted JSON
- Validates issue type from a fixed allowed set:
  `bug, feature_request, documentation, question, other`
- Handles long issue bodies by truncation
- Validates and repairs malformed JSON responses

---

## 🛡️ Edge Cases Handled

✔ Invalid repo  
✔ Issue number doesn’t exist  
✔ Repo has issues disabled  
✔ No comments  
✔ Very long bodies (truncation applied)  
✔ GitHub API errors (404/410)  
✔ Backend down detection  
✔ LLM malformed JSON recovery  

---

## ⚙️ Setup

### 1️⃣ Clone the repo

```

git clone [https://github.com/divyansh2034/seedling-lab-assignment.git](https://github.com/divyansh2034/seedling-lab-assignment.git)
cd seedling-lab-assignment

```

### 2️⃣ Install dependencies

```

pip install -r requirements.txt

```

### 3️⃣ Add free API keys to `.env`

```

HF_TOKEN=your_huggingface_token
GITHUB_TOKEN=your_github_pat

```

### 4️⃣ Run backend

```

cd backend
uvicorn main:app --reload

```

### 5️⃣ Run frontend

```

cd ../frontend
streamlit run streamlit_app.py

```

---

## ⚡ Speed & Performance

- FastAPI for API efficiency
- HuggingFace inference API for free LLM usage
- Truncation to reduce model tokens
- Streamlit optimized UI

---

## 🧪 Tested With

- Valid repos
- Invalid repos
- Empty comments
- Very long issue bodies
- Repos where issues are disabled
- Token errors
- No internet / backend down

---

## 🎯 Rubric Alignment

### ✔ Problem Solving & AI Acumen
- Clean multi-step prompt
- Strong JSON structure enforcement
- Handles malformed responses

### ✔ Code Quality
- Clear folder structure
- Well commented (even for beginners)
- Uses `.env` + `.gitignore` properly

### ✔ Speed & Efficiency
- Lightweight architecture
- Optimal libraries

### ✔ Communication & Initiative
- History tab
- JSON download
- Backend health check
- Repository metadata

---

## 🚀 Bonus Features

- Re-run past issues
- Download JSON output
- Enhanced UI styling
- Repo metadata fetch

---

## 📈 Future Enhancements

- Batch issue analysis
- DB storage
- Slack/Jira integration

---

## 👤 Author

**Divyansh Agarwal**  
Software developer & AI enthusiast.

- GitHub: https://github.com/divyansh2034
- Email: divyansha.cs22@rvce.edu.in

---

## 🔐 Security Notes

- `.env` and `venv/` are ignored using `.gitignore`
- Tokens never committed to repo
- Tokens rotated when exposed

---

## 🪪 License

MIT License
