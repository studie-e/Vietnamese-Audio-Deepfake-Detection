"""Model quantization + pruning utilities for AASIST / Wav2Vec.

Usage examples:
  python vispoofdb/scripts/quantize.py --model-path path/to/checkpoint.pt --n-benchmark 50

Outputs a JSON summary and saves the recommended optimized model.
"""
import os
import time
import copy
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


# Defaults
MODEL_PATH_DEFAULT = "best_model.pt"
RESULTS_FILE_DEFAULT = "quantize_results.json"
SAMPLE_RATE = 16000
AUDIO_LEN = SAMPLE_RATE * 4
N_BENCHMARK = 100
TARGET_SIZE_MB = 100
TARGET_LATENCY_MS = 1000


def get_model_size_mb(model) -> float:
    tmp = "_tmp_model_size.pt"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024 ** 2)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return round(size, 2)


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.nonzero().size(0) for p in model.parameters())
    return {
        "total": total,
        "nonzero": nonzero,
        "sparsity": round(1 - nonzero / total, 4) if total > 0 else 0,
    }


def measure_inference_time(model, device: str, n_runs: int = N_BENCHMARK) -> dict:
    model.eval()
    dummy = torch.randn(1, AUDIO_LEN).to(device)
    # warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
    }


def quick_accuracy(model, test_pairs: list, device: str) -> float:
    model.eval()
    correct = 0
    if not test_pairs:
        return None
    with torch.no_grad():
        for audio, label in test_pairs:
            if len(audio) > AUDIO_LEN:
                audio = audio[:AUDIO_LEN]
            else:
                audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
            x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            if pred == label:
                correct += 1
    return correct / len(test_pairs)


def apply_dynamic_quantization(model) -> nn.Module:
    quantized = torch.quantization.quantize_dynamic(
        model.cpu(),
        qconfig_spec={nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8,
        inplace=False,
    )
    return quantized


def apply_pruning(model, amount: float = 0.3) -> nn.Module:
    pruned = copy.deepcopy(model)
    for name, module in pruned.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return pruned


def apply_prune_then_quantize(model, prune_amount: float = 0.3) -> nn.Module:
    pruned = apply_pruning(model, amount=prune_amount)
    quantized = apply_dynamic_quantization(pruned)
    return quantized


def export_torchscript(model, save_path: str = "model_scripted.pt") -> str:
    model.eval().cpu()
    try:
        scripted = torch.jit.script(model)
        scripted.save(save_path)
        return save_path
    except Exception:
        dummy = torch.randn(1, AUDIO_LEN)
        traced = torch.jit.trace(model, dummy)
        traced.save(save_path)
        return save_path


def benchmark_all(model_orig, test_pairs: list, device: str, n_benchmark: int = N_BENCHMARK) -> dict:
    results = {}
    configs = [
        ("Original", model_orig),
        ("INT8 Quantization", apply_dynamic_quantization(model_orig)),
        ("Pruning 30%", apply_pruning(model_orig, 0.30)),
        ("Pruning 50%", apply_pruning(model_orig, 0.50)),
        ("Pruning 30% + INT8", apply_prune_then_quantize(model_orig, 0.30)),
    ]
    for name, model_variant in configs:
        model_variant.eval()
        size = get_model_size_mb(model_variant)
        latency = measure_inference_time(model_variant, device, n_runs=n_benchmark)
        params = count_parameters(model_variant)
        acc = quick_accuracy(model_variant, test_pairs, device) if test_pairs else None
        results[name] = {
            "size_mb": size,
            "latency": latency,
            "params": params,
            "accuracy": acc,
            "meets_target": (size < TARGET_SIZE_MB and latency["mean_ms"] < TARGET_LATENCY_MS),
        }
    return results


def save_optimized_model(model, name: str, results: dict):
    safe_name = name.lower().replace(" ", "_").replace("%", "p")
    save_path = f"model_{safe_name}.pt"
    torch.save(model.state_dict(), save_path)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=MODEL_PATH_DEFAULT)
    parser.add_argument("--results-file", type=str, default=RESULTS_FILE_DEFAULT)
    parser.add_argument("--n-benchmark", type=int, default=50)
    parser.add_argument("--use-test-samples", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print("Model checkpoint not found:", model_path)
        return

    print("Loading model checkpoint...")
    try:
        import sys
        sys.path.insert(0, ".")
        # Prefer local AASIST package if available
        try:
            from AASIST.models.baseline import Full_AASIST_Model as AASISTModel
        except Exception:
            # fallback to generic import used previously
            from models.AASIST import AASISTModel

        model = AASISTModel()
        ckpt = torch.load(str(model_path), map_location="cpu")
        state = (ckpt.get("model_state_dict") or ckpt.get("model") or ckpt)
        model.load_state_dict(state)
        model.eval().cpu()
    except Exception as e:
        print("Failed to load AASIST model:", e)
        print("If the AASIST package is in a different folder, provide --model-path and ensure imports are resolvable or set --model-root to AASIST root.")
        return

    test_pairs = []
    if args.use_test_samples:
        try:
            import pandas as pd
            import soundfile as sf
            meta = Path('vispoofdb/data/clean_data/metadata.csv')
            if meta.exists():
                df = pd.read_csv(meta)
                sample = df[df['split'].str.contains('test', na=False)].sample(50, random_state=42)
                for _, row in sample.iterrows():
                    try:
                        audio, sr = sf.read(row['file_path'], dtype='float32', always_2d=False)
                        label = 0 if row['label'] == 'real' else 1
                        test_pairs.append((audio, label))
                    except Exception:
                        pass
        except Exception as e:
            print('Failed to load test samples:', e)

    print('Running benchmarks...')
    results = benchmark_all(model, test_pairs, device='cpu', n_benchmark=args.n_benchmark)

    # export torchscript of original
    _ = export_torchscript(model, 'model_scripted.pt')

    # save recommended model (prune30+int8)
    optimized = apply_prune_then_quantize(model, prune_amount=0.30)
    saved = save_optimized_model(optimized, 'pruning_30_int8', results)
    print('Saved optimized model at', saved)

    # write results json
    out = {}
    for k, v in results.items():
        out[k] = {
            'size_mb': v['size_mb'],
            'latency_ms': v['latency']['mean_ms'],
            'sparsity': v['params']['sparsity'],
            'accuracy': v['accuracy'],
            'meets_target': v['meets_target'],
        }
    with open(args.results_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('Wrote results to', args.results_file)


if __name__ == '__main__':
    main()
