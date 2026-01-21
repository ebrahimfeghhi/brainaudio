import argparse
import os
from datasets import load_dataset
from tqdm import tqdm


def stream_to_file(dataset_name, output_file, target_size_gb, subset=None, text_column='text', append=False, skip=0):
    """
    Stream a dataset to a text file.

    Args:
        dataset_name: HuggingFace dataset name
        output_file: Output file path
        target_size_gb: Target file size in GB
        subset: Dataset subset/config (optional)
        text_column: Column name containing text
        append: If True, append to existing file instead of overwriting
        skip: Number of samples to skip (useful for resuming)
    """
    mode = "a" if append else "w"

    # Get existing file size if appending
    existing_bytes = 0
    if append and os.path.exists(output_file):
        existing_bytes = os.path.getsize(output_file)
        print(f"\nAppending to existing file ({existing_bytes / (1024**3):.2f} GB)")

    print(f"Streaming from {dataset_name}" + (f" ({subset})" if subset else ""))
    print(f"Target size to add: {target_size_gb} GB")
    print(f"Output: {output_file} ({'append' if append else 'overwrite'})")
    if skip > 0:
        print(f"Skipping first {skip:,} samples")

    # Load dataset in streaming mode
    if subset:
        ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)

    target_bytes = target_size_gb * 1024 * 1024 * 1024
    current_bytes = 0

    with open(output_file, mode, encoding="utf-8") as f:
        progress_bar = tqdm(desc="Bytes Downloaded", unit="B", unit_scale=True, total=target_bytes)

        for i, sample in enumerate(ds):
            # Skip samples if requested
            if i < skip:
                if i % 1_000_000 == 0 and i > 0:
                    print(f"  Skipped {i:,} samples...")
                continue

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
                print(f"\n  Stopped at sample {i:,}")
                break

        progress_bar.close()

    total_bytes = existing_bytes + current_bytes
    print(f"Finished. Added {current_bytes / (1024**3):.2f} GB")
    print(f"Total file size: {total_bytes / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Download OpenSubtitles dataset")
    parser.add_argument("--size", type=float, default=20.0, help="Target size in GB to download")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--skip", type=int, default=0, help="Number of samples to skip (for resuming)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    output_dir = "./lm_training_data"
    os.makedirs(output_dir, exist_ok=True)

    output_file = args.output or os.path.join(output_dir, "raw_subtitles.txt")

    # Using 'dim/opensubtitles_clean_v1' which is a Parquet-based mirror
    # that works with modern datasets library (no trust_remote_code needed)
    stream_to_file(
        dataset_name="dim/opensubtitles_clean_v1",
        output_file=output_file,
        target_size_gb=args.size,
        text_column='text',
        append=args.append,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
