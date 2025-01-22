# RealVista-Analytics
An advanced real estate data analytics chatbot, to perform transactional, visual, comparative, descriptive, predictive and reasoning based comprehensive analysis with only natural laguage query.

This Streamlit application allows users to interact with structured data using both Langchain and PandasAI. The app provides functionalities for uploading custom data, analyzing it, and generating insights using OpenAI's GPT models.

## Features

- **Data Source Selection**: Users can choose to use default data or upload custom CSV/Excel files.
- **Dataset Summary**: Provides a summary of the dataset including the number of rows, columns, and available columns.
- **Usage Ideas**: Offers suggestions on how the dataset can be used for various analyses.
- **Query Analysis**: Users can input queries related to the dataset, and the app will classify the intent and provide relevant analysis using Langchain or PandasAI.
- **Visualization**: Generates visualizations based on user queries.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Shashank545/RealVista-Analytics/tree/main
    ```
2. Navigate to the project directory:
    ```bash
    cd <project_directory>
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Data Source Selection**: Choose between using default data or uploading a custom CSV/Excel file.
2. **Dataset Summary**: View a summary of the dataset, including the number of rows, columns, and available columns.
3. **Query Analysis**: Enter a query related to the dataset. The app will classify the intent and provide relevant analysis using Langchain or PandasAI.
4. **Visualization**: Generate visualizations based on user queries.

## Environment Variables

Create a `.env` file in the project directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key

```


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- Streamlit
- Langchain
- PandasAI
- OpenAI