# Security Guide

## API Keys & Secrets Management

This project uses secure secret management to protect API keys. Never commit API keys to version control.

## Secure Setup

### Local Development
1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Add your actual API keys to `.streamlit/secrets.toml`
3. The `.streamlit/secrets.toml` file is in `.gitignore` and will not be committed

### Streamlit Cloud Deployment
1. Go to your Streamlit Cloud app settings
2. Navigate to Settings > Secrets
3. Add each API key as a key-value pair:
   ```
   TIINGO_API_KEY = "your_key_here"
   FMP_API_KEY = "your_key_here"
   ALPHA_VANTAGE_API_KEY = "your_key_here"
   GEMINI_API_KEY = "your_key_here"
   NEWS_API_KEY = "your_key_here"
   ```

## How Secrets Are Loaded

The app uses a `get_secret()` function that:
1. First tries to load from `st.secrets` (Streamlit Cloud or local `.streamlit/secrets.toml`)
2. Falls back to environment variables if secrets are not found
3. Never hardcodes API keys in the source code

## Security Checklist

Before making your repository public:
- [x] `.streamlit/secrets.toml` is in `.gitignore`
- [x] No API keys are hardcoded in any `.py` files
- [x] `secrets.toml.example` exists as a template (without real keys)
- [x] All sensitive files are excluded from version control

## If You Accidentally Committed Secrets

1. Immediately rotate/regenerate all exposed API keys
2. Remove the file from git history:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .streamlit/secrets.toml" \
     --prune-empty --tag-name-filter cat -- --all
   ```
3. Force push (warn collaborators first):
   ```bash
   git push origin --force --all
   ```

## Required API Keys

| Service | Purpose | Get API Key |
|---------|---------|-------------|
| Tiingo | Financial fundamentals data | https://api.tiingo.com/ |
| FMP | Company profiles & financial data | https://site.financialmodelingprep.com/ |
| Alpha Vantage | Stock data & ticker search | https://www.alphavantage.co/ |
| Gemini | AI-powered analysis | https://makersuite.google.com/ |
| NewsAPI | News sentiment analysis | https://newsapi.org/ |

## Verify Security

Check if any secrets are in your git history:
```bash
git log --all --full-history --source -- .streamlit/secrets.toml
```

If this returns nothing, you're safe.
