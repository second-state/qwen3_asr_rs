use anyhow::Result;
use crate::tensor::{DType, Device, Tensor};

/// Whisper-style mel spectrogram feature extractor.
///
/// Parameters match the Qwen3-ASR preprocessor config:
/// - n_fft = 400
/// - hop_length = 160
/// - num_mel_bins = 128
/// - sample_rate = 16000
pub struct WhisperFeatureExtractor {
    n_fft: usize,
    hop_length: usize,
    num_mel_bins: usize,
    sample_rate: u32,
    mel_filters: Tensor, // (num_mel_bins, n_fft/2 + 1)
}

impl WhisperFeatureExtractor {
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        num_mel_bins: usize,
        sample_rate: u32,
        device: Device,
    ) -> Self {
        let mel_filters =
            create_mel_filterbank(num_mel_bins, n_fft, sample_rate, 0.0, sample_rate as f64 / 2.0)
                .to_device(device);

        Self {
            n_fft,
            hop_length,
            num_mel_bins,
            sample_rate,
            mel_filters,
        }
    }

    /// Extract log-mel spectrogram features from audio samples.
    ///
    /// Matches HuggingFace WhisperFeatureExtractor._torch_extract_fbank_features:
    /// 1. STFT with center=True, pad_mode="reflect"
    /// 2. Remove last frame (magnitudes[..., :-1])
    /// 3. Apply mel filterbank and log normalization
    ///
    /// Input: f32 samples at self.sample_rate (16kHz)
    /// Output: (num_mel_bins, num_frames) tensor
    pub fn extract(&self, samples: &[f32], device: Device) -> Result<Tensor> {
        // Pad samples to the next multiple of hop_length to ensure clean frame count.
        let padded_len = ((samples.len() + self.hop_length - 1) / self.hop_length) * self.hop_length;
        let mut padded_samples = samples.to_vec();
        padded_samples.resize(padded_len, 0.0);

        let waveform = Tensor::from_slice_f32(&padded_samples)
            .to_dtype(DType::Float32)
            .to_device(device);

        // Create Hann window
        let window = Tensor::hann_window(self.n_fft as i64, device);

        // Center padding: pad waveform with n_fft//2 reflected samples on each side.
        let pad = (self.n_fft / 2) as i64;
        let waveform = waveform.unsqueeze(0).unsqueeze(0); // (1,1,N) for reflection_pad1d
        let waveform = waveform.reflection_pad1d(&[pad, pad]).squeeze_dim(0).squeeze_dim(0);

        // Compute STFT (no center, since we already padded manually)
        let stft = waveform.stft(
            self.n_fft as i64,           // n_fft
            self.hop_length as i64,      // hop_length
            self.n_fft as i64,           // win_length (defaults to n_fft)
            &window,                     // window
            false,                       // normalized
            true,                        // onesided
            true,                        // return_complex
        );

        // Compute power spectrogram: |STFT|^2
        // stft shape: (n_fft/2+1, num_frames)
        let magnitudes = stft.abs().square();

        // Remove last frame to match Python: magnitudes = magnitudes[..., :-1]
        let num_frames = magnitudes.size()[1];
        let magnitudes = magnitudes.narrow(1, 0, num_frames - 1);

        // Apply mel filterbank: (num_mel_bins, n_fft/2+1) @ (n_fft/2+1, num_frames)
        let mel_spec = self.mel_filters.matmul(&magnitudes);

        // Log-mel spectrogram with Whisper-style normalization
        let log_mel = mel_spec.clamp_min(1e-10).log10();
        let max_val = log_mel.max();
        let log_mel = log_mel.maximum(&(&max_val - 8.0));
        let log_mel = (&log_mel + 4.0) / 4.0;

        Ok(log_mel)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }
}

/// Create a mel filterbank matrix matching HuggingFace WhisperFeatureExtractor.
///
/// Uses the slopes-based construction with:
/// - Slaney mel scale: linear below 1000 Hz, logarithmic above (same as librosa default)
/// - Correct FFT bin frequencies: freq[j] = j * sr / n_fft
/// - Slaney normalization: filter *= 2 / (f_high - f_low)
///
/// Returns a (num_mel_bins, n_fft/2+1) tensor.
fn create_mel_filterbank(
    num_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    fmin: f64,
    fmax: f64,
) -> Tensor {
    let n_freqs = n_fft / 2 + 1;
    let sr = sample_rate as f64;

    // Slaney mel scale parameters (matches librosa and HuggingFace)
    let f_sp = 200.0 / 3.0; // Hz per mel step in linear region
    let min_log_hz = 1000.0; // break frequency
    let min_log_mel = (min_log_hz - 0.0) / f_sp; // mel value at break
    let logstep = (6.4_f64).ln() / 27.0; // step size in log region

    let hz_to_mel = |f: f64| -> f64 {
        if f < min_log_hz {
            f / f_sp
        } else {
            min_log_mel + (f / min_log_hz).ln() / logstep
        }
    };

    let mel_to_hz = |m: f64| -> f64 {
        if m < min_log_mel {
            f_sp * m
        } else {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Equally spaced mel filter edge frequencies
    let filter_freqs: Vec<f64> = (0..num_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (num_mels + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    // FFT bin center frequencies (matching np.fft.rfftfreq)
    let all_freqs: Vec<f64> = (0..n_freqs)
        .map(|j| j as f64 * sr / n_fft as f64)
        .collect();

    // Frequency differences between adjacent mel filter edges
    let f_diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    // Construct triangular filters using slopes method (matches HF/librosa exactly)
    let mut filters = vec![0.0f32; num_mels * n_freqs];

    for j in 0..n_freqs {
        for i in 0..num_mels {
            let down = (all_freqs[j] - filter_freqs[i]) / f_diff[i];
            let up = (filter_freqs[i + 2] - all_freqs[j]) / f_diff[i + 1];
            let val = down.min(up).max(0.0);
            filters[i * n_freqs + j] = val as f32;
        }
    }

    // Slaney normalization
    for i in 0..num_mels {
        let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
        for j in 0..n_freqs {
            filters[i * n_freqs + j] *= enorm as f32;
        }
    }

    Tensor::from_slice_f32(&filters).reshape(&[num_mels as i64, n_freqs as i64])
}
