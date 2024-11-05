import soundfile as sf
import torch
import librosa
import numpy as np
import onnxruntime as ort
from pathlib import Path
from argparse import ArgumentParser
from separate import *

def main():
    parser = ArgumentParser()
    parser.add_argument("--file", type=Path, required=True, help="Source audio path")
    parser.add_argument("-o", "--output", type=Path, default=Path("separated"), help="Output folder") 
    parser.add_argument("-m", "--model_path", type=Path, required=True, help="MDX Net ONNX Model path")
    
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    provider = 'CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'
    model = ort.InferenceSession(str(args.model_path), providers=[provider])

    separator_args = {
        "denoise": True,
        "margin": 44100,
        "chunks": 15,
        "n_fft": 6144,
        "dim_t": 8,
        "dim_f": 2048,
    }

    predictor = Separator(separator_args)

    mix, rate = librosa.load(str(args.file), mono=False, sr=44100)
    if mix.ndim == 1:
        mix = np.stack([mix, mix])

    bgm_segments = []

    for skip, segment in predictor.segment(mix).items():
        spek = predictor.preprocess(segment)

        if separator_args["denoise"]:
            spec_pred = (
                - model.run(None, {"input": -spek})[0] * 0.5
                + model.run(None, {"input": spek})[0] * 0.5
            )
        else:
            spec_pred = model.run(None, {"input": spek})[0]

        bgm_segments.append(predictor.postprocess(skip, spec_pred))

    bgm = np.concatenate(bgm_segments, axis=1)
    bgm = bgm[:, :mix.shape[1]]

    vocal = mix - bgm

    stem_name = args.file.stem
    sf.write(args.output / f"{stem_name}_vocal.wav", vocal.T, rate)
    sf.write(args.output / f"{stem_name}_no_vocal.wav", bgm.T, rate)

if __name__ == "__main__":
    main()