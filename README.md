## How to Run Reveil Bot Detection API - Backend

### 1. Install Dependencies

```cmd
cd backend
pip install -r requirements.txt
```

### 2. Create `.env` File

Create `.env` in the `backend/` folder:

Generate a secret key:

```cmd
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

```env
SECRET_KEY=paste-it-here
```

### 3. Initialize Database

```cmd
python createTables.py
```

### 4. Start Server

```cmd
uvicorn app.main:app --reload
```

### 5. Access API

- **API Docs**: http://127.0.0.1:8000/docs
- **API**: http://127.0.0.1:8000

---

**that's about it**