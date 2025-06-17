# Setup Guide for PDF Assistant with Automated Scoring

## Environment Configuration

To secure your API keys and configure the application, follow these steps:

### 1. Create a .env file

Create a `.env` file in the root directory of your project with the following content:

```env
# Google API Configuration
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# Application Configuration (optional - defaults will be used if not set)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_MEMORY_MESSAGES=10
MAX_FILE_SIZE_MB=50
```

### 2. Get Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Replace `your_google_api_key_here` in your `.env` file with the actual API key

### 3. Security Notes

- The `.env` file is already included in `.gitignore` to prevent accidentally committing your API keys
- Never share your `.env` file or commit it to version control
- Keep your API keys secure and rotate them regularly

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run AES.py
```

## Configuration Options

You can customize the following settings in your `.env` file:

- `CHUNK_SIZE`: Size of text chunks for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_MEMORY_MESSAGES`: Maximum chat history messages (default: 10)
- `MAX_FILE_SIZE_MB`: Maximum PDF file size in MB (default: 50)

## Troubleshooting

If you get an error about the API key not being set, make sure:
1. Your `.env` file exists in the project root
2. The `GOOGLE_API_KEY` variable is set correctly
3. There are no extra spaces or quotes around the API key 