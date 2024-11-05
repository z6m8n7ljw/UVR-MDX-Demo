
import numpy as np

def numpy_hann(M, sym=True):
    """
    Implements the Hann window function using NumPy.

    Parameters:
    -----------
    M : int
        Window length
    sym : bool
        Whether the window is symmetric. True for symmetric window, False for periodic window.
        
    Returns:
    --------
    w : ndarray
        Hann window coefficients
    """
    if not sym:
        M = M + 1

    n = np.arange(M)
    w = 0.5 * (1 - np.cos(2 * np.pi * n / (M-1)))

    if not sym:
        w = w[:-1]
        
    return w


def numpy_stft(x, n_fft, hop, window):
    """
    Computes the Short-Time Fourier Transform (STFT) of the input signal using NumPy.
    
    Parameters:
        x: Input signal array of shape (channels, samples)
        n_fft: Size of FFT window
        hop: Number of samples between successive frames
        window: Window function to apply to each frame
        
    Returns:
        Complex STFT coefficients as real/imaginary components with shape 
        (channels, freq_bins, time_frames, 2)
    """
    if x.ndim == 1:
        x = x[None, :]
    num_channels, _ = x.shape
    
    pad_size = (n_fft // 2, n_fft // 2)
    x_padded = np.pad(x, ((0, 0), pad_size), mode='reflect')
    num_frames = (x_padded.shape[1] - n_fft) // hop + 1
    
    result = np.zeros((num_channels, n_fft // 2 + 1, num_frames), dtype=np.complex64)
    
    for ch in range(num_channels):
        frames = np.lib.stride_tricks.as_strided(
            x_padded[ch],
            shape=(num_frames, n_fft),
            strides=(hop * x_padded[ch].strides[0], x_padded[ch].strides[0])
        )
        
        windowed_frames = frames * window
        fft_result = np.fft.rfft(windowed_frames, n=n_fft)
        result[ch] = fft_result.T
    
    return np.stack([result.real, result.imag], axis=-1)


def numpy_istft(x, n_fft, hop, window):
    """
    Computes the Inverse Short-Time Fourier Transform (ISTFT) to reconstruct the time-domain signal.
    
    Parameters:
        x: STFT coefficients array of shape (channels, freq_bins, time_frames, 2)
        n_fft: Size of FFT window
        hop: Number of samples between successive frames
        window: Window function to apply to each frame
        
    Returns:
        Reconstructed time-domain signal of shape (channels, samples)
    """
    x_complex = x[..., 0] + 1j * x[..., 1]
    
    num_channels, _, num_frames = x_complex.shape
    expected_signal_len = (num_frames - 1) * hop + n_fft

    result = np.zeros((num_channels, expected_signal_len), dtype=np.float32)
    window_buffer = np.zeros((num_channels, expected_signal_len), dtype=np.float32)
    
    for ch in range(num_channels):
        ifft_frames = np.fft.irfft(x_complex[ch].T, n=n_fft)

        ifft_frames *= window

        for frame in range(num_frames):
            start = frame * hop
            end = start + n_fft
            result[ch, start:end] += ifft_frames[frame]
            window_buffer[ch, start:end] += window ** 2

    result = result / (window_buffer + 1e-6)

    pad = n_fft // 2
    result = result[:, pad:-pad]
    
    return result


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
        # self.window = scipy.signal.windows.hann(self.n_fft, sym=False)
        self.window = numpy_hann(self.n_fft, sym=False)
        self.target_name = target_name
        
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        
        # self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.freq_pad = np.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.n = L // 2

    def stft(self, x):
        x = x.reshape(-1, self.chunk_size)  # [B, L]
        # x = torch.stft(
        #     x,
        #     n_fft=self.n_fft,
        #     hop_length=self.hop,
        #     window=self.window,
        #     center=True,
        #     return_complex=True,
        # )

        # def compute_stft(segment):
        #     f, t, Zxx = scipy.signal.stft(
        #         segment,
        #         nperseg=self.n_fft,
        #         noverlap=self.n_fft - self.hop,
        #         window=self.window,
        #         return_onesided=True,
        #         boundary='zeros',
        #     )
        #     Zxx *= (np.float32(self.n_fft) / 2.0)
        #     return Zxx

        # Zxx = np.apply_along_axis(compute_stft, 1, x)
        # Zxx_real = Zxx.real.astype(np.float32)
        # Zxx_imag = Zxx.imag.astype(np.float32)
        # x = np.stack((Zxx_real, Zxx_imag), axis=-1)  # [B, N, T, 2]
        x = numpy_stft(x, self.n_fft, self.hop, self.window)
        # x = x.permute([0, 3, 1, 2])
        x = np.transpose(x, [0, 3, 1, 2])  # [B, 2, N, T]
        # x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        x = x.reshape(-1, self.dim_c, self.n_bins, self.dim_t)  # [B, C, N, T]
        return x[:, :, : self.dim_f]

    # Inversed Short-time Fourier transform (STFT).
    def istft(self, x, freq_pad=None):
        # freq_pad = (
        #     self.freq_pad.repeat([x.shape[0], 1, 1, 1])
        #     if freq_pad is None
        #     else freq_pad
        # )
        if freq_pad is None:
            freq_pad = np.tile(self.freq_pad, (x.shape[0], 1, 1, 1))
        # x = torch.cat([x, freq_pad], -2)
        x = np.concatenate([x, freq_pad], axis=-2)
        c = 4 * 2 if self.target_name == "*" else 2
        new_shape = [-1, c, 2, self.n_bins, self.dim_t]
        x = x.reshape(new_shape)
        # x = x.permute([0, 2, 3, 1])
        x = x.reshape([-1, 2, self.n_bins, self.dim_t])
        x = np.transpose(x, [0, 2, 3, 1])
        # x = x.contiguous()
        # x = torch.view_as_complex(x)
        # x = x[..., 0] + 1j * x[..., 1]
        # x = torch.istft(
        #     x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        # )
        # _, x = scipy.signal.istft(
        #     x, nperseg=self.n_fft, noverlap=self.n_fft - self.hop, window=self.window, input_onesided=True
        # )
        # x /= (np.float32(self.n_fft) / 2.0)
        x = numpy_istft(x, self.n_fft, self.hop, self.window)

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

        mix_p = np.zeros((2, n_sample + 2 * self.trim + self.pad), dtype=cmix.dtype)
        mix_p[:, self.trim:self.trim + n_sample] = cmix

        total_chunks = (n_sample + self.pad) // gen_size
        mix_waves = np.empty((total_chunks, 2, trans.chunk_size), dtype=cmix.dtype)
        for i in range(total_chunks):
            start = i * gen_size
            mix_waves[i] = mix_p[:, start:start + trans.chunk_size]

        # mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32)
        mix_waves = mix_waves.astype(np.float32)
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
        tar_waves_trimmed = tar_waves[:, :, self.trim:-self.trim]
        tar_signal = tar_waves_trimmed.transpose(1, 0, 2).reshape(2, -1)[:, :-self.pad]

        start = 0 if mix == 0 else self.args["margin"]
        end = None if mix == self.last_skip else -self.args["margin"]
        
        if self.args["margin"] == 0:
            end = None

        return np.ascontiguousarray(tar_signal[:, start:end])