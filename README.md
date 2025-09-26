# KAG - Knowledge Augmented Generation

AI-powered PDF question answering system with text highlighting capabilities.

## Requirements

- Docker & Docker Compose
- OpenAI API Key

## Quick Start

1. **Set up environment**:

   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

2. **Start the application**:

   ```bash
   docker-compose up --build -d
   ```

3. **Access the web interface**:

   Open `index.html` in your browser

4. **Stop the application**:

   ```bash
   docker-compose down
   ```

## Usage

- Upload PDF documents via the web interface
- Ask questions about your documents
- View highlighted text passages that support the answers
