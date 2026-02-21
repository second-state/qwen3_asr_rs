use anyhow::Result;
use tch::{Kind, Tensor};

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
        device: tch::Device,
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
    /// Input: f32 samples at self.sample_rate (16kHz)
    /// Output: (num_mel_bins, num_frames) tensor
    pub fn extract(&self, samples: &[f32], device: tch::Device) -> Result<Tensor> {
        let waveform = Tensor::from_slice(samples)
            .to_kind(Kind::Float)
            .to_device(device);

        // Create Hann window
        let window = Tensor::hann_window(self.n_fft as i64, (Kind::Float, device));

        // Compute STFT
        // tch::Tensor::stft returns complex tensor (real, imag) of shape (..., freq_bins, frames)
        let stft = waveform.stft(
            self.n_fft as i64,           // n_fft
            Some(self.hop_length as i64), // hop_length
            None,                        // win_length (defaults to n_fft)
            Some(&window),               // window
            false,                       // normalized
            true,                        // onesided
            true,                        // return_complex
            false,                       // align_to_window
        );

        // Compute power spectrogram: |STFT|^2
        // stft shape: (n_fft/2+1, num_frames)
        let magnitudes = stft.abs().square();

        // Apply mel filterbank: (num_mel_bins, n_fft/2+1) @ (n_fft/2+1, num_frames)
        let mel_spec = self.mel_filters.matmul(&magnitudes);

        // Log-mel spectrogram with Whisper-style normalization
        let log_mel = mel_spec.clamp_min(1e-10).log10();
        let max_val = log_mel.max();
        let log_mel = log_mel.maximum(&(&max_val - 8.0));
        let log_mel = (&log_mel + 4.0) / 4.0;

        Ok(log_mel)
    }

    /// Get the number of output frames for a given number of input samples.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        // With center=false (which stft default uses in this context):
        // num_frames = (num_samples - n_fft) / hop_length + 1
        // But Whisper uses center=True padding, so:
        // num_frames = num_samples / hop_length + 1 (approximately)
        num_samples / self.hop_length + 1
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }
}

/// Create a mel filterbank matrix using HTK mel scale.
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

    // Convert Hz to mel (HTK formula)
    let hz_to_mel = |f: f64| -> f64 { 2595.0 * (1.0 + f / 700.0).log10() };
    let mel_to_hz = |m: f64| -> f64 { 700.0 * (10.0_f64.powf(m / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Create equally spaced mel points
    let n_points = num_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();

    // Convert mel points to Hz
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&f| (n_fft as f64 + 1.0) * f / sr)
        .collect();

    // Create triangular filters
    let mut filters = vec![0.0f32; num_mels * n_freqs];

    for i in 0..num_mels {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        for j in 0..n_freqs {
            let freq = j as f64;

            if freq >= left && freq < center {
                // Rising slope
                let val = (freq - left) / (center - left);
                filters[i * n_freqs + j] = val as f32;
            } else if freq >= center && freq <= right {
                // Falling slope
                let val = (right - freq) / (right - center);
                filters[i * n_freqs + j] = val as f32;
            }
        }

        // Slaney-style normalization: normalize by filter bandwidth
        let enorm = 2.0 / (hz_points[i + 2] - hz_points[i]);
        for j in 0..n_freqs {
            filters[i * n_freqs + j] *= enorm as f32;
        }
    }

    Tensor::from_slice(&filters).reshape([num_mels as i64, n_freqs as i64])
}
