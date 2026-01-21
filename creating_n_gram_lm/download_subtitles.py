import os
from datasets import load_dataset
from tqdm import tqdm


def stream_to_file(dataset_name, output_file, target_size_gb, subset=None, text_column='text'):
    """
    Stream a dataset to a text file.

    Args:
        dataset_name: HuggingFace dataset name
        output_file: Output file path
        target_size_gb: Target file size in GB
        subset: Dataset subset/config (optional)
        text_column: Column name containing text
    """
    print(f"\nStarting stream for {dataset_name}" + (f" ({subset})" if subset else ""))
    print(f"Target size: {target_size_gb} GB")
    print(f"Output: {output_file}")

    # Load dataset in streaming mode
    if subset:
        ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)

    target_bytes = target_size_gb * 1024 * 1024 * 1024
    current_bytes = 0

    with open(output_file, "w", encoding="utf-8") as f:
        progress_bar = tqdm(desc="Bytes Downloaded", unit="B", unit_scale=True, total=target_bytes)

        for sample in ds:
            try:
                text = sample[text_column]
            except KeyError:
                # Print available keys on first error to help debug
                print(f"KeyError: '{text_column}' not found. Available keys: {sample.keys()}")
                continue

            if not text:
                continue

            # Basic cleaning: normalize whitespace
            clean_text = ' '.join(text.split())

            if not clean_text:
                continue

            line = clean_text + "\n"
            f.write(line)

            line_size = len(line.encode('utf-8'))
            current_bytes += line_size
            progress_bar.update(line_size)

            if current_bytes >= target_bytes:
                break

        progress_bar.close()

    print(f"Finished. Saved {current_bytes / (1024**3):.2f} GB to {output_file}")

def main():
    output_dir = "./lm_training_data"
    os.makedirs(output_dir, exist_ok=True)

    # Download OpenSubtitles (Conversational English) -> 20 GB
    # Using 'dim/opensubtitles_clean_v1' which is a Parquet-based mirror
    # that works with modern datasets library (no trust_remote_code needed)
    # Contains cleaned English subtitles with 'text' column
    stream_to_file(
        dataset_name="dim/opensubtitles_clean_v1",
        output_file=os.path.join(output_dir, "raw_subtitles.txt"),
        target_size_gb=20.0,
        text_column='text',
    )


if __name__ == "__main__":
    main()