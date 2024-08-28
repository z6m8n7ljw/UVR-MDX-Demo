
import numpy as np
import scipy

class ConvTDFNet:
    def __init__(self, target_name, L, dim_f, dim_t, n_fft, hop=1024):
        super(ConvTDFNet, self).__init__()
        self.dim_c = 4
        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        # self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.window = scipy.signal.windows.hann(self.n_fft, sym=False)
        self.target_name = target_name
        
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        
        # self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.freq_pad = np.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.n = L // 2

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])  # [B, L]
        # x = torch.stft(
        #     x,
        #     n_fft=self.n_fft,
        #     hop_length=self.hop,
        #     window=self.window,
        #     center=True,
        #     return_complex=True,
        # )
        output = []
        for i in range(x.shape[0]):
            f, t, Zxx = scipy.signal.stft(
                x[i],
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop,
                window=self.window,
                return_onesided=True,
                boundary='zeros',
            )
            Zxx *= (np.float32(self.n_fft) / 2.0)
            output.append(Zxx)
        x = np.array(output, dtype=np.complex64)   # [B, N, T]
        # x = torch.view_as_real(x)
        x = np.stack((x.real, x.imag), axis=-1)  # [B, N, T, 2]
        # x = x.permute([0, 3, 1, 2])
        x = np.transpose(x, [0, 3, 1, 2])  # [B, 2, N, T]
        # x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])  # [B, C, N, T]
        return x[:, :, : self.dim_f]

    # Inversed Short-time Fourier transform (STFT).
    def istft(self, x, freq_pad=None):
        # freq_pad = (
        #     self.freq_pad.repeat([x.shape[0], 1, 1, 1])
        #     if freq_pad is None
        #     else freq_pad
        # )
        freq_pad = (
            np.tile(self.freq_pad, (x.shape[0], 1, 1, 1))
            if freq_pad is None
            else freq_pad
        )
        # x = torch.cat([x, freq_pad], -2)
        x = np.concatenate([x, freq_pad], axis=-2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        # x = x.permute([0, 2, 3, 1])
        x = np.transpose(x, [0, 2, 3, 1])
        # x = x.contiguous()
        x = np.ascontiguousarray(x)
        # x = torch.view_as_complex(x)
        x = x[..., 0] + 1j * x[..., 1]
        # x = torch.istft(
        #     x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        # )
        _, x = scipy.signal.istft(
            x, nperseg=self.n_fft, noverlap=self.n_fft - self.hop, window=self.window, input_onesided=True
        )
        x /= (np.float32(self.n_fft) / 2.0)
        return x.reshape([-1, c, self.chunk_size])

class Separator:
    def __init__(self, args):
        self.args = args
        self.trans = ConvTDFNet(
            target_name="vocals",
            L=11,
            dim_f=args["dim_f"], 
            dim_t=args["dim_t"], 
            n_fft=args["n_fft"]
        )
    
    def segment(self, mix):
        samples = mix.shape[-1]
        chunk_size = self.args["chunks"] * 44100
        
        assert not self.args["margin"] == 0, "margin cannot be zero!"
        
        if self.args["margin"] > chunk_size:
            self.args["margin"] = chunk_size

        segmented_mix = {}

        if self.args["chunks"] == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else self.args["margin"]
            end = min(skip + chunk_size + self.args["margin"], samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break
        
        self.last_skip = list(segmented_mix.keys())[::-1][0]
        return segmented_mix

    def preprocess(self, cmix):
        n_sample = cmix.shape[1]
        trans = self.trans
        self.trim = trans.n_fft // 2
        gen_size = trans.chunk_size - 2 * self.trim
        self.pad = gen_size - n_sample % gen_size

        mix_p = np.concatenate(
            (np.zeros((2, self.trim)), cmix, np.zeros((2, self.pad)), np.zeros((2, self.trim))), 1
        )

        mix_waves = []
        i = 0
        while i < n_sample + self.pad:
            waves = np.array(mix_p[:, i : i + trans.chunk_size])
            mix_waves.append(waves)
            i += gen_size

        # mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32)
        mix_waves = np.array(mix_waves).astype(np.float32)
        spek = trans.stft(mix_waves)
        self.dims = spek.shape
        # return spek.cpu().numpy()
        return np.ascontiguousarray(spek)

    def postprocess(self, mix, spec_pred):
        trans = self.trans
        tar_waves = trans.istft(spec_pred.reshape(self.dims))
        # tar_signal = (
        #     tar_waves[:, :, self.trim:-self.trim]
        #     .transpose(0, 1)
        #     .reshape(2, -1)
        #     .numpy()[:, :-self.pad]
        # )
        tar_signal = (
            np.transpose(tar_waves[:, :, self.trim:-self.trim], (1, 0, 2))
            .reshape(2, -1) 
            [:, :-self.pad]
        )

        start = 0 if mix == 0 else self.args["margin"]
        end = None if mix == self.last_skip else -self.args["margin"]
        
        if self.args["margin"] == 0:
            end = None

        return tar_signal[:, start:end]