"""Compare logit entropy between Transformer and RNN models."""

import numpy as np

BLANK_IDX = 0

trans_path = "/data2/brain2text/b2t_25/logits/best_chunked_transformer_combined_seed_0/logits_val_chunk:5_context:20.npz"
rnn_path = "/data2/brain2text/b2t_25/logits/pretrained_RNN/logits_val.npz"

trans_npz = np.load(trans_path)
rnn_npz = np.load(rnn_path)

trans_keys = sorted([k for k in trans_npz.keys() if k.startswith("arr_")], key=lambda x: int(x.split("_")[1]))
rnn_keys = sorted([k for k in rnn_npz.keys() if k.startswith("arr_")], key=lambda x: int(x.split("_")[1]))


def compute_entropy(npz, keys, name, n_trials=50):
    all_entropy = []
    all_entropy_nonblank_frames = []
    all_entropy_excl_blank_prob = []
    n_blank = 0
    n_total = 0

    for key in keys[:n_trials]:
        logits = npz[key]
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # All frames entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        all_entropy.extend(entropy.flatten())

        # Non-blank frame analysis
        argmax = np.argmax(probs, axis=-1)
        n_total += len(argmax)
        n_blank += (argmax == BLANK_IDX).sum()

        nonblank_mask = argmax != BLANK_IDX
        if nonblank_mask.sum() > 0:
            probs_nb = probs[nonblank_mask]
            entropy_nb = -np.sum(probs_nb * np.log(probs_nb + 1e-10), axis=-1)
            all_entropy_nonblank_frames.extend(entropy_nb.flatten())

            # Exclude blank probability
            probs_no_blank = probs_nb[:, 1:]
            probs_no_blank = probs_no_blank / probs_no_blank.sum(axis=-1, keepdims=True)
            entropy_excl = -np.sum(probs_no_blank * np.log(probs_no_blank + 1e-10), axis=-1)
            all_entropy_excl_blank_prob.extend(entropy_excl.flatten())

    blank_pct = 100 * n_blank / n_total

    print(f"\n{name}:")
    print(f"  Blank frames: {blank_pct:.1f}%")
    print(f"  All frames entropy:              mean={np.mean(all_entropy):.4f}, std={np.std(all_entropy):.4f}")
    print(f"  Non-blank frames entropy:        mean={np.mean(all_entropy_nonblank_frames):.4f}, std={np.std(all_entropy_nonblank_frames):.4f}")
    print(f"  Non-blank (excl blank prob):     mean={np.mean(all_entropy_excl_blank_prob):.4f}, std={np.std(all_entropy_excl_blank_prob):.4f}")


def compute_entropy_at_temperatures(npz, keys, name, temperatures, n_trials=50):
    """Compute entropy at various temperature scales."""
    results = []

    for temp in temperatures:
        all_entropy_excl_blank = []

        for key in keys[:n_trials]:
            logits = npz[key]
            # Apply temperature before softmax
            scaled_logits = logits / temp
            exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

            # Non-blank frames only
            argmax = np.argmax(probs, axis=-1)
            nonblank_mask = argmax != BLANK_IDX

            if nonblank_mask.sum() > 0:
                probs_nb = probs[nonblank_mask]
                # Exclude blank probability
                probs_no_blank = probs_nb[:, 1:]
                probs_no_blank = probs_no_blank / probs_no_blank.sum(axis=-1, keepdims=True)
                entropy = -np.sum(probs_no_blank * np.log(probs_no_blank + 1e-10), axis=-1)
                all_entropy_excl_blank.extend(entropy.flatten())

        results.append({
            'temp': temp,
            'mean': np.mean(all_entropy_excl_blank),
            'std': np.std(all_entropy_excl_blank)
        })

    print(f"\n{name} - Entropy at different temperatures:")
    print(f"  {'Temp':<8} {'Mean':<12} {'Std':<12}")
    print(f"  {'-'*32}")
    for r in results:
        print(f"  {r['temp']:<8.2f} {r['mean']:<12.4f} {r['std']:<12.4f}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Entropy Comparison: Transformer vs RNN")
    print("=" * 60)
    compute_entropy(trans_npz, trans_keys, "Transformer")
    compute_entropy(rnn_npz, rnn_keys, "RNN")

    print("\n" + "=" * 60)
    print("Entropy at Different Temperatures")
    print("=" * 60)

    temperatures = [0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]

    trans_temp_results = compute_entropy_at_temperatures(trans_npz, trans_keys, "Transformer", temperatures)
    rnn_temp_results = compute_entropy_at_temperatures(rnn_npz, rnn_keys, "RNN", temperatures)

    # Find temperature where Transformer matches RNN's baseline entropy
    rnn_baseline = next(r['mean'] for r in rnn_temp_results if r['temp'] == 1.0)
    print(f"\n" + "=" * 60)
    print(f"RNN baseline entropy (temp=1.0): {rnn_baseline:.4f}")
    print(f"To match RNN entropy, Transformer needs approximately:")
    for r in trans_temp_results:
        if r['mean'] >= rnn_baseline:
            print(f"  Temperature ~{r['temp']:.1f} (entropy={r['mean']:.4f})")
            break
