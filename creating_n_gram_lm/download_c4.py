import argparse
import os
from datasets import load_dataset
from tqdm import tqdm


def stream_to_file(dataset_name, subset, output_file, target_size_gb, text_column='text', append=False, skip=0):
    """
    Streams a dataset from Hugging Face and saves it to a text file
    until the target file size is reached.

    Args:
        append: If True, append to existing file instead of overwriting
        skip: Number of samples to skip (useful for resuming)
    """
    mode = "a" if append else "w"

    # Get existing file size if appending
    existing_bytes = 0
    if append and os.path.exists(output_file):
        existing_bytes = os.path.getsize(output_file)
        print(f"\nAppending to existing file ({existing_bytes / (1024**3):.2f} GB)")

    print(f"Streaming from {dataset_name} ({subset})...")
    print(f"Target size to add: {target_size_gb} GB")
    print(f"Output: {output_file} ({'append' if append else 'overwrite'})")
    if skip > 0:
        print(f"Skipping first {skip:,} samples")

    # Load dataset in streaming mode
    if subset:
        ds = load_dataset(dataset_name, subset, split="train", streaming=True, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)

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

            # Extract text
            try:
                text = sample[text_column]
            except KeyError:
                continue

            # Basic cleaning: flatten newlines to keep 1 sentence per line
            clean_text = text.replace('\n', ' ').strip()

            if not clean_text:
                continue

            line = clean_text + "\n"
            f.write(line)

            # Update size tracking
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
    parser = argparse.ArgumentParser(description="Download C4 dataset")
    parser.add_argument("--size", type=float, default=20.0, help="Target size in GB to download")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--skip", type=int, default=0, help="Number of samples to skip (for resuming)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    output_dir = "./lm_training_data"
    os.makedirs(output_dir, exist_ok=True)

    output_file = args.output or os.path.join(output_dir, "raw_c4.txt")

    stream_to_file(
        dataset_name="allenai/c4",
        subset="en",
        output_file=output_file,
        target_size_gb=args.size,
        append=args.append,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
