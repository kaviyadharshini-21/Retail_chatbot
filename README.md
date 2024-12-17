# Financial Statement Analyzer

## Overview

The Financial Statement Analyzer is a web application designed to assist users in analyzing financial data. It utilizes LangChain and various AI models to provide insights and answers based on user queries related to retail sales data.

## Features

- User-friendly chat interface for querying financial data.
- File upload functionality for CSV files.
- Integration with LangChain for advanced data processing and retrieval.
- Responsive design for optimal viewing on various devices.

## Requirements

To run this application, you need to have the following Python packages installed:

- langchain
- langchain_groq
- langchain_core
- langchain_community
- langchain_google_genai
- langchain_chroma
- Flask

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a `.env` file in the root directory and add your environment variables:

   ```
   GROQ_API_KEY=<your_groq_api_key>
   ```

3. Prepare your dataset:

   Ensure you have the `retail_sales_dataset.csv` file in the `data` directory.

4. Run the database migration script to load the CSV data into SQLite:

   ```bash
   python csv_todb.py
   ```

5. Start the Flask application:

   ```bash
   python app.py
   ```

6. Open your web browser and navigate to `http://127.0.0.1:5000` to access the application.

## Usage

- Use the chat interface to ask questions about the retail sales data.
- You can upload additional CSV files for analysis.
- The application will respond with structured and detailed answers based on the provided data.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
