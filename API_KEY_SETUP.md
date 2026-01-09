# API Key Setup Guide

This project uses Streamlit secrets to securely manage your Google Gemini API key. This eliminates the need to manually enter your API key each time you run the app.

## Quick Setup

1. **Get your API key** (if you don't have one):
   - Visit: https://aistudio.google.com/apikey
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the generated key

2. **Configure secrets.toml**:
   - Open `.streamlit/secrets.toml` in your project
   - Replace `"your_api_key_here"` with your actual API key:
     ```toml
     GEMINI_API_KEY = "AIzaSyC..."  # Your actual key here
     ```
   - Save the file

3. **Restart Streamlit**:
   - The app will automatically load your API key from `secrets.toml`
   - No need to enter it manually in the sidebar!

## Security Notes

- ✅ `.streamlit/secrets.toml` is automatically ignored by Git (see `.gitignore`)
- ✅ Never commit your actual API key to version control
- ✅ Use `secrets.toml.example` as a template for other developers

## Alternative Methods

The app checks for API keys in this priority order:

1. **Streamlit secrets** (`.streamlit/secrets.toml`) - **Recommended**
2. Session state (user input in sidebar)
3. Environment variables (`GEMINI_API_KEY` or `GOOGLE_API_KEY`)

### Using Environment Variables

If you prefer environment variables instead:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Troubleshooting

- **"API Key Required" error**: Make sure you've updated `secrets.toml` with your actual key (not the placeholder)
- **Key not loading**: Ensure the file is located at `.streamlit/secrets.toml` (note the leading dot)
- **Key format**: Make sure your key is in quotes: `GEMINI_API_KEY = "AIza..."`

## File Structure

```
your-project/
├── .streamlit/
│   ├── secrets.toml          # Your actual API key (DO NOT COMMIT)
│   └── secrets.toml.example  # Template file (safe to commit)
├── .gitignore                # Ensures secrets.toml is ignored
└── app_premium.py            # Main app file
```
