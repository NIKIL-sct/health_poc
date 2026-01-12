1. Create a Virtual Environment
Linux / macOS
    python3 -m venv venv
Windows (Command Prompt / PowerShell)
    python -m venv venv

2. Activate the Virtual Environment
Linux / macOS
    source venv/bin/activate
Windows (Command Prompt)
    venv\Scripts\activate
Windows (PowerShell)
    venv\Scripts\Activate.ps1

If PowerShell blocks activation, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

3. Upgrade pip
    pip install --upgrade pip

4. Install Python Dependencies
    pip install -r requirements.txt

5. Running the Service
    Activate the virtual environment (if not already active)

Linux / macOS
    source venv/bin/activate

Windows
    venv\Scripts\activate

Start the application
    python main.py
