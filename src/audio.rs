use anyhow::{Context, Result};

/// Load an audio file, converting to mono f32 at the target sample rate.
///
/// Uses ffmpeg to decode any audio format (MP3, FLAC, AAC, WAV, OGG, etc.).
/// Falls back to hound for WAV files if ffmpeg fails.
pub fn load_audio(path: &str, target_sample_rate: u32) -> Result<Vec<f32>> {
    match load_audio_ffmpeg(path, target_sample_rate) {
        Ok(samples) => Ok(samples),
        Err(e) => {
            tracing::warn!("FFmpeg loading failed ({}), trying WAV fallback", e);
            load_audio_wav(path, target_sample_rate)
        }
    }
}

/// Load audio using ffmpeg (any format).
fn load_audio_ffmpeg(path: &str, target_sample_rate: u32) -> Result<Vec<f32>> {
    ffmpeg_next::init().context("Failed to initialize ffmpeg")?;

    let mut input_ctx =
        ffmpeg_next::format::input(&path).context("Failed to open audio file")?;

    let audio_stream = input_ctx
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .ok_or_else(|| anyhow::anyhow!("No audio stream found in {}", path))?;

    let stream_index = audio_stream.index();
    let codec_params = audio_stream.parameters();

    let codec_ctx = ffmpeg_next::codec::Context::from_parameters(codec_params)
        .context("Failed to create codec context")?;
    let mut decoder = codec_ctx.decoder().audio().context("Failed to create audio decoder")?;

    let mut all_samples: Vec<f32> = Vec::new();
    let mut resampler: Option<ffmpeg_next::software::resampling::Context> = None;

    // Process packets
    for (stream, packet) in input_ctx.packets() {
        if stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        decode_and_resample(
            &mut decoder,
            &mut resampler,
            target_sample_rate,
            &mut all_samples,
        )?;
    }

    // Flush decoder
    decoder.send_eof()?;
    decode_and_resample(
        &mut decoder,
        &mut resampler,
        target_sample_rate,
        &mut all_samples,
    )?;

    // Flush resampler
    if let Some(ref mut res) = resampler {
        flush_resampler(res, &mut all_samples);
    }

    if all_samples.is_empty() {
        anyhow::bail!("No audio samples decoded");
    }

    tracing::info!(
        "Loaded audio via FFmpeg: {} samples ({:.2}s at {}Hz)",
        all_samples.len(),
        all_samples.len() as f64 / target_sample_rate as f64,
        target_sample_rate
    );

    Ok(all_samples)
}

/// Decode frames and resample. Creates the resampler lazily from the first decoded frame.
fn decode_and_resample(
    decoder: &mut ffmpeg_next::decoder::Audio,
    resampler: &mut Option<ffmpeg_next::software::resampling::Context>,
    target_sample_rate: u32,
    samples: &mut Vec<f32>,
) -> Result<()> {
    let target_format = ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Packed);
    let target_layout = ffmpeg_next::ChannelLayout::MONO;

    let mut decoded = ffmpeg_next::frame::Audio::empty();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let already_target = decoded.format() == target_format
            && decoded.channel_layout() == target_layout
            && decoded.rate() == target_sample_rate;

        if already_target {
            // No conversion needed, extract samples directly
            append_f32_samples(&decoded, samples);
            continue;
        }

        // FFmpeg 8.x may return an empty channel layout for some files.
        // Set a default layout on the decoded frame so resampling works.
        if decoded.channel_layout().is_empty() {
            decoded.set_channel_layout(
                ffmpeg_next::ChannelLayout::default(decoded.channels() as i32),
            );
        }

        // Create resampler lazily from the actual decoded frame's properties
        if resampler.is_none() {
            let res = ffmpeg_next::software::resampling::Context::get(
                decoded.format(),
                decoded.channel_layout(),
                decoded.rate(),
                target_format,
                target_layout,
                target_sample_rate,
            )
            .context("Failed to create resampler from decoded frame")?;
            *resampler = Some(res);
        }

        let res = resampler.as_mut().unwrap();
        let mut resampled = ffmpeg_next::frame::Audio::empty();
        res.run(&decoded, &mut resampled)
            .context("Resampling failed")?;
        append_f32_samples(&resampled, samples);
    }
    Ok(())
}

fn flush_resampler(
    resampler: &mut ffmpeg_next::software::resampling::Context,
    samples: &mut Vec<f32>,
) {
    let mut resampled = ffmpeg_next::frame::Audio::empty();
    // Flush remaining buffered samples. Errors are non-fatal here.
    for _ in 0..10 {
        match resampler.flush(&mut resampled) {
            Ok(Some(_delay)) => {
                append_f32_samples(&resampled, samples);
            }
            _ => break,
        }
    }
}

fn append_f32_samples(frame: &ffmpeg_next::frame::Audio, samples: &mut Vec<f32>) {
    let n = frame.samples();
    if n == 0 {
        return;
    }
    let data = frame.data(0);
    let f32_slice =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
    samples.extend_from_slice(f32_slice);
}

/// Load a WAV file and resample to target rate using rubato.
fn load_audio_wav(path: &str, target_sample_rate: u32) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path).context("Failed to open WAV file")?;
    let spec = reader.spec();

    tracing::info!(
        "WAV: {}ch, {}Hz, {:?}, {}bit",
        spec.channels,
        spec.sample_rate,
        spec.sample_format,
        spec.bits_per_sample
    );

    // Read all samples as f32
    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .map(|s| s.unwrap())
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono_samples = if spec.channels > 1 {
        let channels = spec.channels as usize;
        raw_samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw_samples
    };

    // Resample if needed
    let samples = if spec.sample_rate != target_sample_rate {
        resample(&mono_samples, spec.sample_rate, target_sample_rate)?
    } else {
        mono_samples
    };

    tracing::info!(
        "Loaded WAV: {} samples ({:.2}s at {}Hz)",
        samples.len(),
        samples.len() as f64 / target_sample_rate as f64,
        target_sample_rate
    );

    Ok(samples)
}

/// Resample audio using rubato.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, Resampler, WindowFunction};

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        samples.len(),
        1, // mono
    )
    .map_err(|e| anyhow::anyhow!("Failed to create resampler: {}", e))?;

    let result = resampler
        .process(&[samples.to_vec()], None)
        .map_err(|e| anyhow::anyhow!("Resampling failed: {}", e))?;

    Ok(result.into_iter().next().unwrap_or_default())
}
