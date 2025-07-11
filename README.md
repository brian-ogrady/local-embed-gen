# **local-embed-gen**

A Python utility for efficiently generating text embeddings locally at a small to medium scale (up to \~10 million records). This tool is designed to process large CSV files in chunks, generate embeddings using various state-of-the-art models, and save the results in the efficient Apache Parquet format.

## **Features**

* **Memory Efficient**: Processes large files in configurable chunks to keep memory usage low.  
* **Multiple Model Support**: Easily switch between different embedding models, including sparse (e.g., BM25), dense (e.g., Sentence-Transformers), and multi-vector models (e.g., BGE-M3, ColBERT).  
* **Flexible Configuration**: Model parameters and types are managed through a simple JSON configuration file.  
* **Optimized Output**: Saves embeddings to Zstandard-compressed Parquet files for efficient storage and fast subsequent loading.  
* **Hardware Acceleration**: Automatically utilizes available hardware like MPS (on Apple Silicon) for faster processing.

## **Getting Started**

Follow these instructions to get the project set up and running on your local machine.

### **Prerequisites**

* Python 3.11 or higher

### **Installation**

1. **Clone the repository:**  
   git clone https://github.com/brian-ogrady/local-embed-gen.git  
   cd local-embed-gen

2. Install dependencies:  
   The project uses uv for dependency management. You can install the dependencies directly using uv from the pyproject.toml file.  
   uv pip install .

   This will install all necessary packages, including:  
   * polars for data manipulation  
   * fastparquet and pyarrow for working with Parquet files  
   * fastembed, FlagEmbedding, sentence-transformers, and pylate for the various embedding models.

## **Usage**

The primary script for generating embeddings is scripts/generate\_embeddings.py. You can run it from the command line with several options to customize the process.

### **Command-Line Arguments**

| Argument | Type | Required | Default | Description |
| :---- | :---- | :---- | :---- | :---- |
| \--input-file | str | **Yes** | None | Path to the input CSV file containing the text data. |
| \--output-file | str | **Yes** | None | Path for the output Parquet file where embeddings will be saved. |
| \--model-config-path | str | **Yes** | None | Path to the JSON file with model configurations. |
| \--model-name | str | **Yes** | None | The key of the model to use from the config file (e.g., 'bgem3'). |
| \--text-column | str | No | keyword | Name of the column in the input CSV that contains the text to embed. |
| \--chunksize | int | No | 10000 | Number of rows to process in each chunk. Adjust based on your available RAM. |

### **Example Command**

Here is an example of how to run the script to generate embeddings using the BAAI/bge-m3 model:

python scripts/generate\_embeddings.py \\  
    \--input-file ./path/to/your/data.csv \\  
    \--output-file ./output/embeddings.parquet \\  
    \--model-config-path ./configs/supported\_models.json \\  
    \--model-name bgem3 \\  
    \--text-column "product\_description" \\  
    \--chunksize 20000

## **Supported Models**

The models are defined in configs/supported\_models.json. You can easily add or modify model configurations in that file. The currently supported models are:

| Model Key | Library Used | Hugging Face / Model Name | Output Type |
| :---- | :---- | :---- | :---- |
| bm25 | fastembed | Qdrant/bm25 | Sparse |
| bgem3 | flagembedding | BAAI/bge-m3 | Dense, Colbert, Sparse |
| e5large | sentencetransformers | intfloat/multilingual-e5-large | Dense |
| paraphrase | sentencetransformers | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | Dense |
| gte | sentencetransformers | Alibaba-NLP/gte-multilingual-base | Dense |
| qwen | sentencetransformers | Qwen/Qwen3-Embedding-0.6B | Dense |
| gte\_colbert | pylate | lightonai/GTE-ModernColBERT-v1 | Colbert |

### **Output Schema**

The script generates a Parquet file containing the original data from the input CSV along with the new embedding columns. The names of the embedding columns depend on the model used.

* For standard dense models (e.g., e5large), a single column named after the model-name (e.g., e5large) is added.  
* For the bgem3 model, multiple columns are added to store the different types of embeddings: bgem3\_dense\_vecs, bgem3\_colbert\_vecs, bgem3\_sparse\_indices, and bgem3\_sparse\_values.

## **Contributing**

Contributions are welcome\! If you have suggestions for improvements or want to add support for a new model, feel free to open an issue or submit a pull request.

## **License**

This project is currently unlicensed. You are free to use, modify, and distribute it.