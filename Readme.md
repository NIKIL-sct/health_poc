## 1. Create a Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv
```

### Windows (Command Prompt / PowerShell)

```bat
python -m venv venv
```

---

## 2. Activate the Virtual Environment

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows (Command Prompt)

```bat
venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
venv\Scripts\Activate.ps1
```

> **Note:** If PowerShell blocks script execution, run the following command once:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 3. Upgrade pip

```bash
pip install --upgrade pip
```

---

## 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Running the Service

### Activate the Virtual Environment (if not already active)

* **Linux / macOS**

  ```bash
  source venv/bin/activate
  ```

* **Windows**

  ```bat
  venv\Scripts\activate
  ```

### Start the Application

```bash
python main.py
```

---
---
