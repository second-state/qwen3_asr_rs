use std::path::PathBuf;

use gpui::*;
use gpui_platform::application;
use std::sync::Arc;
use qwen3_asr_rs::inference::AsrInference;
use qwen3_asr_rs::capture::AudioCapture;
use qwen3_asr_rs::tensor::Device;


fn find_model_dir() -> Option<PathBuf> {
    let cache = model_dir_cache();
    if cache.join("config.json").exists() {
        return Some(cache);
    }
    let local = PathBuf::from("qwen3_asr_rs/Qwen3-ASR-0.6B");
    if local.join("config.json").exists() {
        return Some(local);
    }
    let local17 = PathBuf::from("qwen3_asr_rs/Qwen3-ASR-1.7B");
    if local17.join("config.json").exists() {
        return Some(local17);
    }
    None
}

fn model_dir_cache() -> PathBuf {
    let home = dirs_fallback();
    home.join(".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0")
}

fn dirs_fallback() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .map(PathBuf::from)
            .or_else(|_| {
                std::env::var("HOMEDRIVE").and_then(|d| {
                    std::env::var("HOMEPATH").map(|p| PathBuf::from(format!("{d}{p}")))
                })
            })
            .unwrap_or_else(|_| PathBuf::from("."))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Toggle,
    Hold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMethod {
    Clipboard,
    DirectInput,
}

struct VoiceInputApp {
    audio: Option<AudioCapture>,
    transcribed_text: SharedString,
    is_recording: bool,
    input_mode: InputMode,
    output_method: OutputMethod,
    asr: Arc<AsrInference>,
}

impl Render for VoiceInputApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .bg(rgb(0x1e1e2e))
            .size_full()
            .gap_4()
            .p_6()
            .on_key_down(cx.listener(Self::on_key_down))
            .on_key_up(cx.listener(Self::on_key_up))
            .child(
                div()
                    .text_xl()
                    .font_weight(FontWeight::BOLD)
                    .text_color(rgb(0xcdd6f4))
                    .child("abrust"),
            )
            .child(
                div()
                    .flex_1()
                    .w_full()
                    .bg(rgb(0x313244))
                    .rounded_lg()
                    .p_4()
                    .text_color(rgb(0xcdd6f4))
                    .text_base()
                    .child(if self.transcribed_text.is_empty() {
                        SharedString::from("文字起こし結果がここに表示されます...")
                    } else {
                        self.transcribed_text.clone()
                    }),
            )
            .child(
                div()
                    .flex()
                    .gap_3()
                    .justify_center()
                    .child(
                        div()
                            .id("record-button")
                            .flex()
                            .gap_2()
                            .items_center()
                            .px_6()
                            .py_3()
                            .rounded_md()
                            .bg(if self.is_recording {
                                rgb(0xf38ba8)
                            } else {
                                rgb(0xa6e3a1)
                            })
                            .text_color(rgb(0x1e1e2e))
                            .font_weight(FontWeight::MEDIUM)
                            .on_click(cx.listener(Self::on_record_click))
                            .child((if self.is_recording {
                                    "■ 停止"
                                } else {
                                    "● 録音"
                                }).to_string()),
                    )
                    .child(
                        div()
                            .id("clear-button")
                            .px_6()
                            .py_3()
                            .rounded_md()
                            .bg(rgb(0x45475a))
                            .text_color(rgb(0xcdd6f4))
                            .font_weight(FontWeight::MEDIUM)
                            .on_click(cx.listener(|this, _event: &ClickEvent, _window, _cx| {
                                this.transcribed_text = SharedString::default();
                            }))
                            .child("クリア"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_3()
                    .justify_center()
                    .child(
                        div()
                            .id("mode-toggle")
                            .px_4()
                            .py_2()
                            .rounded_md()
                            .bg(rgb(0x585b70))
                            .text_color(rgb(0xcdd6f4))
                            .text_sm()
                            .on_click(cx.listener(Self::on_mode_click))
                            .child(SharedString::from(format!("録音: {}", match self.input_mode {
                                InputMode::Toggle => "トグル",
                                InputMode::Hold => "ホールド",
                            }))),
                    )
                    .child(
                        div()
                            .id("output-toggle")
                            .px_4()
                            .py_2()
                            .rounded_md()
                            .bg(rgb(0x585b70))
                            .text_color(rgb(0xcdd6f4))
                            .text_sm()
                            .on_click(cx.listener(Self::on_output_click))
                            .child(SharedString::from(format!("出力: {}", match self.output_method {
                                OutputMethod::Clipboard => "クリップボード",
                                OutputMethod::DirectInput => "直接入力",
                            }))),
                    )
                    .child(
                        div()
                            .id("copy-button")
                            .px_4()
                            .py_2()
                            .rounded_md()
                            .bg(rgb(0x585b70))
                            .text_color(rgb(0xcdd6f4))
                            .text_sm()
                            .on_click(cx.listener(Self::on_copy_click))
                            .child(SharedString::from("コピー")),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x585b70))
                    .child(SharedString::from(
                        "Ctrl+T: トグル  |  Ctrl+H: ホールド  |  Ctrl+C: コピー  |  Ctrl+Shift+C: クリア  |  Ctrl+O: 出力切替",
                    )),
            )
    }
}

impl VoiceInputApp {
    fn on_record_click(
        &mut self,
        _event: &ClickEvent,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) {
        self.toggle_recording();
    }

    fn on_mode_click(
        &mut self,
        _event: &ClickEvent,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) {
        if self.is_recording {
            self.stop_recording();
        }
        self.input_mode = match self.input_mode {
            InputMode::Toggle => InputMode::Hold,
            InputMode::Hold => InputMode::Toggle,
        };
    }

    fn on_output_click(
        &mut self,
        _event: &ClickEvent,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) {
        self.output_method = match self.output_method {
            OutputMethod::Clipboard => OutputMethod::DirectInput,
            OutputMethod::DirectInput => OutputMethod::Clipboard,
        };
    }

    fn on_copy_click(
        &mut self,
        _event: &ClickEvent,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) {
        self.output_text();
    }

    fn on_key_down(&mut self, event: &KeyDownEvent, _window: &mut Window, _cx: &mut Context<Self>) {
        let key = event.keystroke.key.as_str();
        let mods = event.keystroke.modifiers;

        if mods.control {
            match key {
                "t" => {
                    self.input_mode = InputMode::Toggle;
                    if !self.is_recording {
                        self.start_recording();
                        self.is_recording = true;
                    }
                }
                "h" => {
                    self.input_mode = InputMode::Hold;
                    if !self.is_recording {
                        self.start_recording();
                        self.is_recording = true;
                    }
                }
                "c" => {
                    if mods.shift {
                        self.transcribed_text = SharedString::default();
                    } else {
                        self.output_text();
                    }
                }
                "o" => {
                    self.output_method = match self.output_method {
                        OutputMethod::Clipboard => OutputMethod::DirectInput,
                        OutputMethod::DirectInput => OutputMethod::Clipboard,
                    };
                }
                _ => {}
            }
        }
    }

    fn on_key_up(&mut self, event: &KeyUpEvent, _window: &mut Window, _cx: &mut Context<Self>) {
        if self.input_mode == InputMode::Hold && self.is_recording {
            let key = event.keystroke.key.as_str();
            if key == "h" {
                self.stop_recording();
                self.is_recording = false;
            }
        }
    }

    fn toggle_recording(&mut self) {
        if self.is_recording {
            self.stop_recording();
        } else {
            self.start_recording();
        }
        self.is_recording = !self.is_recording;
    }

    fn start_recording(&mut self) {
        match AudioCapture::new() {
            Ok(mut capture) => {
                if let Err(e) = capture.start() {
                    self.transcribed_text = SharedString::from(format!("録音開始エラー: {e}"));
                    self.is_recording = false;
                    return;
                }
                self.audio = Some(capture);
            }
            Err(e) => {
                self.transcribed_text = SharedString::from(format!("デバイスエラー: {e}"));
                self.is_recording = false;
            }
        }
    }

    fn stop_recording(&mut self) {
        if let Some(mut capture) = self.audio.take() {
            capture.stop();
            let samples = capture.drain_buffer();
            let sample_rate = capture.sample_rate();

            if samples.is_empty() {
                self.transcribed_text = SharedString::from("録音データが空です");
                return;
            }

            self.transcribed_text = SharedString::from("文字起こし処理中...");

            let samples_16k = if sample_rate != 16000 {
                resample(&samples, sample_rate, 16000)
            } else {
                samples
            };

            match self.asr.transcribe_samples(&samples_16k, Some("ja")) {
                Ok(result) => {
                    self.transcribed_text = SharedString::from(result.text);
                }
                Err(e) => {
                    self.transcribed_text = SharedString::from(format!("ASRエラー: {e}"));
                }
            }
        }
    }

    fn output_text(&self) {
        if self.transcribed_text.is_empty() {
            return;
        }
        match self.output_method {
            OutputMethod::Clipboard => {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    clipboard.set_text(self.transcribed_text.as_ref()).ok();
                }
            }
            OutputMethod::DirectInput => {
                #[cfg(target_os = "linux")]
                {
                    use std::io::Write;
                    use std::process::Command;
                    if let Ok(mut child) = Command::new("xdotool")
                        .args(["type", "--clearmodifiers", "--file", "-"])
                        .stdin(std::process::Stdio::piped())
                        .spawn()
                    {
                        if let Some(mut stdin) = child.stdin.take() {
                            let _ = stdin.write_all(self.transcribed_text.as_bytes());
                        }
                        let _ = child.wait();
                    }
                }
                #[cfg(target_os = "windows")]
                {
                    // TODO: SendInput / keybd_event simulation
                }
                #[cfg(target_os = "macos")]
                {
                    // TODO: CGEvent simulation
                }
            }
        }
    }
}

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_idx = (i as f64 * ratio) as usize;
        let frac = (i as f64 * ratio) - src_idx as f64;
        let a = samples.get(src_idx).copied().unwrap_or(0.0);
        let b = samples.get(src_idx + 1).copied().unwrap_or(a);
        output.push(a as f64 * (1.0 - frac) + b as f64 * frac);
    }
    let max_val = output
        .iter()
        .map(|s| s.abs() as f32)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(1.0);
    if max_val > 0.0 {
        for s in &mut output {
            *s /= max_val as f64;
        }
    }
    output.into_iter().map(|s| s as f32).collect()
}

fn main() {
    let model_path = find_model_dir().expect("モデルディレクトリが見つかりません。models/qwen3-asr-0.6b に配置するか、HFキャッシュを確認してください");

    let device = {
        #[cfg(feature = "tch-backend")]
        {
            if tch::Cuda::is_available() {
                Device::Gpu(0)
            } else {
                Device::Cpu
            }
        }
        #[cfg(not(feature = "tch-backend"))]
        Device::Cpu
    };

    let asr = AsrInference::load(&model_path, device)
        .expect("モデルの読み込みに失敗しました");
    let asr = std::sync::Arc::new(asr);

    application().run(move |cx: &mut App| {
        cx.activate(true);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds {
                    origin: point(px(0.), px(0.)),
                    size: size(px(520.), px(420.)),
                })),
                ..Default::default()
            },
            |_window, cx| {
                cx.new(|_cx| VoiceInputApp {
                    audio: None,
                    transcribed_text: SharedString::default(),
                    is_recording: false,
                    input_mode: InputMode::Toggle,
                    output_method: OutputMethod::Clipboard,
                    asr: asr.clone(),
                })
            },
        )
        .unwrap();
    });
}
