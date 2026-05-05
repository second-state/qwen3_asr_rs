use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

pub struct AudioCapture {
    stream: Option<cpal::Stream>,
    buffer: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
}

impl AudioCapture {
    pub fn new() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("no input device available")?;

        let config = device
            .default_input_config()
            .map_err(|e| format!("failed to get default input config: {e}"))?;

        let sample_rate = config.sample_rate();
        let buffer = Arc::new(Mutex::new(Vec::new()));

        Ok(Self {
            stream: None,
            buffer,
            sample_rate,
        })
    }

    pub fn start(&mut self) -> Result<(), String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("no input device available")?;

        let config = device
            .default_input_config()
            .map_err(|e| format!("failed to get default input config: {e}"))?;

        let buffer = self.buffer.clone();
        let err_fn = |err| eprintln!("audio stream error: {err}");

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut buf) = buffer.lock() {
                        buf.extend_from_slice(data);
                    }
                },
                err_fn,
                None,
            ),
            _ => {
                return Err("unsupported sample format — expected F32".into());
            }
        }
        .map_err(|e| format!("failed to build input stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("failed to play stream: {e}"))?;
        self.stream = Some(stream);
        Ok(())
    }

    pub fn stop(&mut self) {
        self.stream.take();
    }

    pub fn drain_buffer(&self) -> Vec<f32> {
        let mut buf = self.buffer.lock().unwrap();
        std::mem::take(&mut *buf)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
