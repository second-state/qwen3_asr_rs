use anyhow::{Context, Result};
use std::path::Path;
use tch::{Device, Tensor};

use crate::audio;
use crate::audio_encoder::AudioEncoder;
use crate::config::AsrConfig;
use crate::layers::compute_mrope_cos_sin;
use crate::mel::WhisperFeatureExtractor;
use crate::text_decoder::{create_causal_mask, KvCache, TextDecoder};
use crate::tokenizer::{
    AsrTokenizer, AUDIO_PAD_TOKEN_ID, ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID,
};
use crate::weights;

const MEL_SAMPLE_RATE: u32 = 16000;

/// ASR inference engine.
pub struct AsrInference {
    audio_encoder: AudioEncoder,
    text_decoder: TextDecoder,
    mel_extractor: WhisperFeatureExtractor,
    tokenizer: AsrTokenizer,
    config: AsrConfig,
    device: Device,
}

impl AsrInference {
    /// Load model from directory containing config.json, model.safetensors, tokenizer.json
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        tracing::info!("Loading model from {:?}", model_dir);

        // Load config
        let config =
            AsrConfig::from_file(&model_dir.join("config.json")).context("Failed to load config")?;

        // Load weights (supports both single-file and sharded safetensors)
        let all_weights =
            weights::load_model_weights(model_dir, device).context("Failed to load weights")?;

        tracing::info!("Loaded {} weight tensors", all_weights.len());

        // Load audio encoder
        tracing::info!("Loading audio encoder...");
        let audio_encoder = AudioEncoder::load(
            &all_weights,
            "thinker.audio_tower",
            &config.thinker_config.audio_config,
            device,
        )
        .context("Failed to load audio encoder")?;

        // Load text decoder
        tracing::info!("Loading text decoder...");
        let text_decoder = TextDecoder::load(
            &all_weights,
            "thinker.model",
            &config.thinker_config.text_config,
        )
        .context("Failed to load text decoder")?;

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer =
            AsrTokenizer::from_dir(model_dir).context("Failed to load tokenizer")?;

        // Create mel feature extractor
        let mel_extractor = WhisperFeatureExtractor::new(
            400,  // n_fft
            160,  // hop_length
            config.thinker_config.audio_config.num_mel_bins,
            MEL_SAMPLE_RATE,
            device,
        );

        tracing::info!("Model loaded successfully");

        Ok(Self {
            audio_encoder,
            text_decoder,
            mel_extractor,
            tokenizer,
            config,
            device,
        })
    }

    /// Transcribe an audio file.
    ///
    /// Returns the transcribed text and detected language.
    pub fn transcribe(&self, audio_path: &str, language: Option<&str>) -> Result<TranscribeResult> {
        // Step 1: Load and preprocess audio (ffmpeg decodes any format to mono 16kHz f32)
        tracing::info!("Loading audio from {}", audio_path);
        let samples = audio::load_audio(audio_path, MEL_SAMPLE_RATE)?;

        // Step 2: Compute mel spectrogram
        let mel = self.mel_extractor.extract(&samples, self.device)?;
        let num_mel_frames = mel.size()[1] as usize;
        tracing::info!("Mel spectrogram: {} frames", num_mel_frames);

        // Step 3: Run audio encoder
        let audio_embeds = self.audio_encoder.forward(&mel);
        let num_audio_tokens = audio_embeds.size()[0] as usize;
        tracing::info!("Audio encoder: {} tokens", num_audio_tokens);

        // Step 4: Build input token sequence
        let (input_ids, audio_positions) =
            self.build_prompt(num_audio_tokens, language)?;
        let seq_len = input_ids.len();

        // Step 5: Build embeddings with audio injection
        let input_tensor =
            Tensor::from_slice(&input_ids).to_device(self.device);
        let mut hidden_states = self.text_decoder.embed(&input_tensor).unsqueeze(0);

        // Replace audio_pad positions with audio encoder embeddings
        for (embed_idx, &seq_pos) in audio_positions.iter().enumerate() {
            let audio_embed = audio_embeds.get(embed_idx as i64);
            hidden_states = hidden_states.slice_scatter(
                &audio_embed.unsqueeze(0).unsqueeze(0),
                1,
                seq_pos as i64,
                seq_pos as i64 + 1,
                1,
            );
        }

        // Step 6: Build MRoPE position IDs (all 3 dims sequential)
        let position_ids = self.build_position_ids(&input_ids);

        let text_config = &self.config.thinker_config.text_config;
        let (cos, sin) = compute_mrope_cos_sin(
            &position_ids,
            text_config.head_dim,
            text_config.rope_theta,
            &text_config.mrope_section(),
            text_config.mrope_interleaved(),
            self.device,
        );

        // Step 7: Prefill - run all prompt tokens through decoder
        let mask = create_causal_mask(seq_len as i64, 0, self.device);
        let mut kv_cache = KvCache::new(text_config.num_hidden_layers);

        let logits = self.text_decoder.forward(
            &hidden_states,
            &cos,
            &sin,
            &mut kv_cache,
            Some(&mask),
        );

        // Step 8: Autoregressive generation
        let mut generated_ids: Vec<i64> = Vec::new();
        let max_new_tokens = 4096;
        let eos_token_ids = vec![ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];

        // Get the last token's logits for the first prediction
        let mut next_logits = logits.narrow(1, seq_len as i64 - 1, 1).squeeze_dim(1);

        let mut current_pos = position_ids[0].len();

        for _ in 0..max_new_tokens {
            // Greedy decoding
            let next_token = next_logits.argmax(-1, false).int64_value(&[0]);

            // Check for EOS
            if eos_token_ids.contains(&next_token) {
                break;
            }

            generated_ids.push(next_token);

            // Prepare next step
            let next_input = Tensor::from_slice(&[next_token]).to_device(self.device);
            let next_hidden = self.text_decoder.embed(&next_input).unsqueeze(0);

            // Position IDs for new token (all 3 dims same)
            let new_pos_ids: [Vec<i64>; 3] = [
                vec![current_pos as i64],
                vec![current_pos as i64],
                vec![current_pos as i64],
            ];

            let (new_cos, new_sin) = compute_mrope_cos_sin(
                &new_pos_ids,
                text_config.head_dim,
                text_config.rope_theta,
                &text_config.mrope_section(),
                text_config.mrope_interleaved(),
                self.device,
            );

            let total_len = kv_cache.seq_len();
            let mask = create_causal_mask(1, total_len, self.device);

            next_logits = self.text_decoder.forward(
                &next_hidden,
                &new_cos,
                &new_sin,
                &mut kv_cache,
                Some(&mask),
            );
            next_logits = next_logits.squeeze_dim(1);

            current_pos += 1;
        }

        // Step 9: Parse output
        tracing::info!("Generated {} tokens", generated_ids.len());
        let raw_text = self.tokenizer.decode(&generated_ids)?;
        tracing::debug!("Raw output: {:?}", raw_text);
        let (language_detected, transcription) = parse_asr_output(&raw_text, language.is_some());

        Ok(TranscribeResult {
            text: transcription,
            language: language_detected,
            raw_output: raw_text,
        })
    }

    /// Build the prompt token sequence for ASR.
    ///
    /// Official format:
    /// PREFIX: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    /// AUDIO:  <|audio_pad|> * num_audio_tokens
    /// SUFFIX: <|audio_end|><|im_end|>\n<|im_start|>assistant\n
    ///
    /// Returns: (token_ids, audio_pad_positions)
    fn build_prompt(
        &self,
        num_audio_tokens: usize,
        language: Option<&str>,
    ) -> Result<(Vec<i64>, Vec<usize>)> {
        let mut tokens: Vec<i64> = vec![
            151644, // <|im_start|>
            8948,   // system
            198,    // \n
            151645, // <|im_end|>
            198,    // \n
            151644, // <|im_start|>
            872,    // user
            198,    // \n
            151669, // <|audio_start|>
        ];

        // Record audio pad positions
        let audio_start_pos = tokens.len();
        for _ in 0..num_audio_tokens {
            tokens.push(AUDIO_PAD_TOKEN_ID);
        }
        let audio_positions: Vec<usize> =
            (audio_start_pos..audio_start_pos + num_audio_tokens).collect();

        // Suffix tokens
        tokens.extend_from_slice(&[
            151670, // <|audio_end|>
            151645, // <|im_end|>
            198,    // \n
            151644, // <|im_start|>
        ]);

        if let Some(lang) = language {
            tokens.push(77091); // assistant
            tokens.push(198);   // \n
            let prefix = format!("language {}", capitalize_first(lang));
            tokens.extend(self.tokenizer.encode(&prefix)?);
        } else {
            tokens.push(77091); // assistant
            tokens.push(198);   // \n
        }

        Ok((tokens, audio_positions))
    }

    /// Build 3D MRoPE position IDs for the input sequence.
    ///
    /// All 3 dims use the same sequential positions [0, 1, 2, ..., N-1].
    /// This matches the Python model's get_rope_index which simply uses
    /// attention_mask.cumsum(-1) - 1 for all 3 dims.
    fn build_position_ids(
        &self,
        input_ids: &[i64],
    ) -> [Vec<i64>; 3] {
        let seq_len = input_ids.len();
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        [positions.clone(), positions.clone(), positions]
    }
}

/// Result of ASR transcription.
pub struct TranscribeResult {
    pub text: String,
    pub language: String,
    pub raw_output: String,
}

/// Parse the model's raw output to extract language and transcription.
///
/// Expected format: "language {Language}<asr_text>{transcription}"
/// When language is forced, format is just: "{transcription}"
fn parse_asr_output(raw: &str, language_forced: bool) -> (String, String) {
    if language_forced {
        return ("forced".to_string(), raw.trim().to_string());
    }

    let raw = raw.trim();

    if let Some(rest) = raw.strip_prefix("language ") {
        if let Some(asr_pos) = rest.find("<asr_text>") {
            let lang = rest[..asr_pos].trim().to_string();
            let text = rest[asr_pos + "<asr_text>".len()..].trim().to_string();
            return (lang, text);
        }
        // Fallback: split at first non-alpha char after language name
        let mut lang_end = 0;
        for (i, c) in rest.char_indices() {
            if c.is_whitespace() || !c.is_alphabetic() {
                lang_end = i;
                break;
            }
            lang_end = i + c.len_utf8();
        }
        if lang_end > 0 {
            let lang = rest[..lang_end].to_string();
            let text = rest[lang_end..].trim().to_string();
            return (lang, text);
        }
    }

    ("unknown".to_string(), raw.to_string())
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
