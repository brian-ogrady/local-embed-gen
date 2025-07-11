import argparse
import json
from typing import Any, Dict, List, Union

import polars as pl
import pyarrow.parquet as pq
from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel
from pylate.models import ColBERT
from sentence_transformers import SentenceTransformer


def get_model(model_config: Dict[str, Any]):
    if model_config["type"] == "fastembed":
        model = SparseTextEmbedding(model_name=model_config["model_name"])
    elif model_config["type"] == "flagembedding":
        model = BGEM3FlagModel(model_config["model_name"], device="mps")
    elif model_config["type"] == "pylate":
        model = ColBERT(model_config["model_name"], device="mps")
    else:
        model = SentenceTransformer(
            model_config["model_name"], 
            device="mps", 
            **model_config.get("model_kwargs", {})
        )
    return model


def generate_embeddings(
    model, 
    model_config: Dict[str, Any], 
    texts: Union[List[str], str]
):
    if isinstance(texts, str):
        texts = [texts]
    if model_config["type"] == "fastembed":
        embeddings = list(model.embed(texts))
        return [
            {
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist()
            } for emb in embeddings
        ]
    return model.encode(texts, **model_config.get("encode_kwargs", {}))
        

def main(args):
    """Main function to run the embedding generation process."""
    print(f"Loading model configuration from: {args.model_config_path}")
    with open(args.model_config_path, "r") as f:
        model_configs = json.load(f)[0]
        model_config = model_configs[args.model_name]
    
    model = get_model(model_config)
    id_key = model_config.get("id_key", args.text_column)

    print(f"Reading CSV '{args.input_file}' in chunks of {args.chunksize}...")
    chunk_iterator = pl.read_csv_batched(
        args.input_file, 
        batch_size=args.chunksize,
    )

    total_rows_processed = 0
    i = 0

    writer = None

    try:
        while (chunk := chunk_iterator.next_batches(1)) is not None:

            chunk = chunk[0]
            i += 1
            print(f"--- Processing Chunk {i} ---")

            output_chunk = chunk.drop_nulls(subset=id_key)

            if output_chunk.is_empty():
                print("Skipping chunk with no text to embed.")
                continue

            texts_to_embed = output_chunk[id_key].cast(pl.String).to_list()

            print(f"Encoding {len(texts_to_embed)} texts...")
            embeddings = generate_embeddings(model, model_config, texts_to_embed)
            
            if model_config["type"] == "flagembedding" and isinstance(embeddings, dict):
                lexical_weights = embeddings["lexical_weights"]
                sparse_indices = [[int(k) for k in d.keys()] for d in lexical_weights]
                sparse_values = [[float(v) for v in d.values()] for d in lexical_weights]

                colbert_embeddings = [emb.tolist() for emb in embeddings["colbert_vecs"]]

                output_chunk = output_chunk.with_columns(
                    pl.Series("bgem3_dense_vecs", embeddings["dense_vecs"]),
                    pl.Series("bgem3_colbert_vecs", colbert_embeddings),
                    pl.Series("bgem3_sparse_indices", sparse_indices),
                    pl.Series("bgem3_sparse_values", sparse_values)
                )
            else:
                output_chunk = output_chunk.with_columns(
                    pl.Series(args.model_name, embeddings)
                )

            print(f"Writing {len(output_chunk)} rows to Parquet file: {args.output_file}")
            arrow_table = output_chunk.to_arrow()

            if writer is None:
                print(f"Creating new Parquet file: {args.output_file}")
                writer = pq.ParquetWriter(
                    args.output_file,
                    arrow_table.schema,
                    compression="zstd",
                    compression_level=3,
                )
            
            writer.write_table(arrow_table)
            
            total_rows_processed += len(output_chunk)
            print(f"Chunk {i+1} complete. Total rows processed: {total_rows_processed}")

    finally:
        if writer:
            writer.close()
            print(f"\n✅ Embedding generation complete! Output saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text embeddings from a CSV file using various models."
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="keyword",
        help="Name of the column containing the text to embed."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path for the output Parquet file."
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        required=True,
        help="Path to the JSON file with model configurations."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="The key of the model to use from the config file (e.g., 'bgem3')."
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="Number of rows to process in each chunk."
    )
    
    args = parser.parse_args()
    main(args)