use leptos::prelude::*;
use serde::Deserialize;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, HtmlImageElement, HtmlInputElement};
use xorae_core::preprocess::{rgba_to_bits, PreprocessOptions};
use xorae_core::Model;

#[derive(Clone, Debug, PartialEq, Deserialize)]
struct ModelEntry {
    id: String,
    name: String,
    path: String,
    #[serde(default)]
    description: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
struct ModelIndex {
    models: Vec<ModelEntry>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
struct SampleEntry {
    name: String,
    path: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
struct SampleIndex {
    samples: Vec<SampleEntry>,
}

type Inference = (Vec<u8>, Vec<u8>, Vec<u8>);

#[component]
pub fn App() -> impl IntoView {
    let models = RwSignal::new(Vec::<ModelEntry>::new());
    let model = RwSignal::new(None::<Model>);
    let status = RwSignal::new("loading model index…".to_string());
    let samples = RwSignal::new(Vec::<SampleEntry>::new());
    let asset_prefix = RwSignal::new(String::new());

    let image_url = RwSignal::new(None::<String>);
    let original = RwSignal::new(Vec::<u8>::new());
    let key = RwSignal::new(Vec::<u8>::new());
    let recon = RwSignal::new(Vec::<u8>::new());

    let diff = Memo::new(move |_| {
        let a = original.get();
        let b = recon.get();
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| u8::from(x != y))
            .collect::<Vec<u8>>()
    });

    let ber = Memo::new(move |_| {
        let a = original.get();
        if a.is_empty() {
            return 0.0f32;
        }
        let b = recon.get();
        let errors = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        errors as f32 / a.len() as f32
    });

    let key_hex = Memo::new(move |_| {
        let bits = key.get();
        if bits.is_empty() {
            return String::new();
        }
        let mut hex = String::new();
        for chunk in bits.chunks(4) {
            let mut value = 0u8;
            for (i, bit) in chunk.iter().enumerate() {
                value |= (bit & 1) << i;
            }
            hex.push(char::from_digit(value as u32, 16).unwrap_or('0'));
        }
        hex
    });

    Effect::new(move |_| {
        spawn_local(async move {
            match fetch_json_with_prefix::<ModelIndex>("models/index.json").await {
                Ok((index, prefix)) => {
                    if index.models.is_empty() {
                        status.set("models/index.json is empty".into());
                        return;
                    }
                    asset_prefix.set(prefix.to_string());
                    let first = index.models[0].clone();
                    models.set(index.models);
                    load_model_at(
                        first.name.clone(),
                        first.path.clone(),
                        prefix.to_string(),
                        model,
                        status,
                    )
                    .await;
                    if let Ok(index) =
                        fetch_json::<SampleIndex>(&format!("{prefix}samples/index.json")).await
                    {
                        if let Some(first_sample) = index.samples.first() {
                            let path = format!("{prefix}{}", first_sample.path);
                            if let Some(loaded) = model.get() {
                                if let Ok((bits, k, r)) = url_to_bits(&path, &loaded).await {
                                    original.set(bits);
                                    key.set(k);
                                    recon.set(r);
                                    image_url.set(Some(path));
                                }
                            }
                        }
                        samples.set(index.samples);
                    }
                }
                Err(e) => status.set(format!(
                    "Could not load models/index.json ({e}). Train and export a model first."
                )),
            }
        });
    });

    let on_model_change = move |ev| {
        let index = event_target_value(&ev).parse::<usize>().unwrap_or(0);
        if let Some(entry) = models.get().get(index).cloned() {
            let prefix = asset_prefix.get();
            spawn_local(load_model_at(entry.name, entry.path, prefix, model, status));
        }
    };

    let on_file = move |ev: web_sys::Event| {
        let Some(input) = ev
            .target()
            .and_then(|target| target.dyn_into::<HtmlInputElement>().ok())
        else {
            return;
        };
        let Some(files) = input.files() else {
            return;
        };
        let Some(file) = files.get(0) else {
            return;
        };
        let Some(loaded) = model.get() else {
            status.set("Load a model first.".into());
            return;
        };
        spawn_local(async move {
            status.set("running model…".into());
            match file_to_bits(file, &loaded).await {
                Ok(((bits, k, r), url)) => {
                    original.set(bits);
                    key.set(k);
                    recon.set(r);
                    image_url.set(Some(url));
                    status.set("done".into());
                }
                Err(e) => status.set(e),
            }
        });
    };

    let key_grid = move || {
        key.get()
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let class = if value == 1 {
                    "keybit on"
                } else {
                    "keybit off"
                };
                view! {
                    <button
                        class=class
                        title=format!("toggle key bit {i}")
                        on:click=move |_| {
                            key.update(|bits| {
                                if i < bits.len() {
                                    bits[i] ^= 1;
                                }
                            });
                            if let Some(loaded) = model.get() {
                                match loaded.decode_key(&key.get()) {
                                    Ok(r) => recon.set(r),
                                    Err(e) => status.set(e),
                                }
                            }
                        }
                    >
                        {format!("{i}")}
                    </button>
                }
            })
            .collect_view()
    };

    let sample_buttons = move || {
        samples
            .get()
            .into_iter()
            .map(|sample| {
                let path = format!("{}{}", asset_prefix.get(), sample.path);
                view! {
                    <button
                        class="sample"
                        on:click=move |_| {
                            let path = path.clone();
                            let Some(loaded) = model.get() else {
                                status.set("Load a model first.".into());
                                return;
                            };
                            spawn_local(async move {
                                status.set(format!("running {path}…"));
                                match url_to_bits(&path, &loaded).await {
                                    Ok((bits, k, r)) => {
                                        original.set(bits);
                                        key.set(k);
                                        recon.set(r);
                                        image_url.set(Some(path.clone()));
                                        status.set("done".into());
                                    }
                                    Err(e) => status.set(e),
                                }
                            });
                        }
                    >
                        {sample.name}
                    </button>
                }
            })
            .collect_view()
    };

    view! {
        <main class="app">
            <header class="hero">
                <div>
                    <p class="eyebrow">"deterministic-ai"</p>
                    <h1>"XOR autoencoder"</h1>
                    <p class="lede">
                        "A genetic algorithm evolves a wired Boolean circuit. The encoder squishes a "
                        {move || model.get().map(|m| format!("{}×{}", m.config.image_size, m.config.image_size)).unwrap_or_else(|| "16×16".into())}
                        " image into a key; the decoder rebuilds it from the key alone."
                    </p>
                </div>
                <div class="model-picker">
                    <label for="model">"model"</label>
                    <select id="model" on:change=on_model_change>
                        {move || models.get().into_iter().enumerate().map(|(i, entry)| view! {
                            <option value=i.to_string() title=entry.description data-id=entry.id>{entry.name}</option>
                        }).collect_view()}
                    </select>
                </div>
            </header>

            <p class="status">{move || status.get()}</p>

            {move || model.get().map(|m| view! {
                <section class="modelcard">
                    <div><span>"gate set"</span><strong>{format!("{:?}", m.config.gate_set)}</strong></div>
                    <div><span>"key"</span><strong>{format!("{} bits", m.config.key_bits)}</strong></div>
                    <div><span>"gates"</span><strong>{format!("{}", m.encoder.gates.len() + m.decoder.gates.len())}</strong></div>
                    <div><span>"val BER"</span><strong>{format!("{:.2}%", m.metrics.val_ber * 100.0)}</strong></div>
                    <div><span>"test BER"</span><strong>{format!("{:.2}%", m.metrics.test_ber * 100.0)}</strong></div>
                    <div><span>"seed"</span><strong>{format!("{}", m.config.seed)}</strong></div>
                </section>
            })}

            <section class="controls">
                <label class="file-input">
                    <span>"upload an image"</span>
                    <input type="file" accept="image/*" on:change=on_file />
                </label>
                <div class="samples">{sample_buttons}</div>
            </section>

            <section class="panels">
                <article class="panel">
                    <h2>"input"</h2>
                    <div class="preview-wrap">
                        {move || image_url.get().map(|url| view! {
                            <img class="preview" src=url alt="input" />
                        })}
                    </div>
                    <BitGrid bits=original.into() diff=diff.into() />
                </article>

                <article class="panel">
                    <h2>"key"</h2>
                    <p class="keyhex">{move || key_hex.get()}</p>
                    <p class="hint">"Click a bit to flip it and re-run the decoder."</p>
                    <div class="keygrid">{key_grid}</div>
                </article>

                <article class="panel">
                    <h2>"reconstruction"</h2>
                    <BitGrid bits=recon.into() diff=diff.into() />
                    <p class="metric">
                        "bit error rate "
                        <strong>{move || format!("{:.2}%", ber.get() * 100.0)}</strong>
                    </p>
                    <p class="hint">"Red cells differ from the binarized input."</p>
                </article>
            </section>
        </main>
    }
}

#[component]
fn BitGrid(bits: Signal<Vec<u8>>, diff: Signal<Vec<u8>>) -> impl IntoView {
    let style = Memo::new(move |_| {
        let count = bits.get().len();
        let side = (count as f64).sqrt().round() as usize;
        format!("grid-template-columns: repeat({}, 1fr);", side.max(1))
    });
    let cells = move || {
        let values = bits.get();
        let changes = diff.get();
        values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let changed = changes.get(i).copied().unwrap_or(0) == 1;
                let class = match (value & 1 == 1, changed) {
                    (true, true) => "bit on changed",
                    (true, false) => "bit on",
                    (false, true) => "bit off changed",
                    (false, false) => "bit off",
                };
                view! { <div class=class></div> }
            })
            .collect_view()
    };
    view! { <div class="bitgrid" style=move || style.get()>{cells}</div> }
}

async fn load_model_at(
    name: String,
    path: String,
    prefix: String,
    model: RwSignal<Option<Model>>,
    status: RwSignal<String>,
) {
    let url = format!("{prefix}{path}");
    status.set(format!("loading {name}…"));
    match fetch_json::<Model>(&url).await {
        Ok(loaded) => {
            status.set(format!(
                "loaded {name} (val BER {:.2}%)",
                loaded.metrics.val_ber * 100.0
            ));
            model.set(Some(loaded));
        }
        Err(e) => status.set(format!("failed to load {url}: {e}")),
    }
}

async fn fetch_json_with_prefix<T: serde::de::DeserializeOwned>(
    path: &str,
) -> Result<(T, &'static str), String> {
    match fetch_json::<T>(path).await {
        Ok(value) => Ok((value, "")),
        Err(first_error) => match fetch_json::<T>(&format!("public/{path}")).await {
            Ok(value) => Ok((value, "public/")),
            Err(_) => Err(first_error),
        },
    }
}

async fn fetch_json<T: serde::de::DeserializeOwned>(url: &str) -> Result<T, String> {
    let response = gloo_net::http::Request::get(url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.ok() {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    response.json::<T>().await.map_err(|e| e.to_string())
}

async fn file_to_bits(file: web_sys::File, model: &Model) -> Result<(Inference, String), String> {
    let url = web_sys::Url::create_object_url_with_blob(&file)
        .map_err(|_| "could not create an object URL".to_string())?;
    let image = load_image(&url).await?;
    let inference = image_to_bits(&image, model)?;
    Ok((inference, url))
}

async fn url_to_bits(url: &str, model: &Model) -> Result<Inference, String> {
    let image = load_image(url).await?;
    image_to_bits(&image, model)
}

async fn load_image(url: &str) -> Result<HtmlImageElement, String> {
    let image = HtmlImageElement::new().map_err(|_| "could not create image element")?;
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let image_for_load = image.clone();
        let onload = Closure::once_into_js(move || {
            let _ = resolve.call0(&JsValue::NULL);
        });
        image_for_load.set_onload(Some(onload.unchecked_ref::<js_sys::Function>()));

        let image_for_error = image.clone();
        let onerror = Closure::once_into_js(move || {
            let _ = reject.call0(&JsValue::NULL);
        });
        image_for_error.set_onerror(Some(onerror.unchecked_ref::<js_sys::Function>()));
    });
    image.set_src(url);
    JsFuture::from(promise)
        .await
        .map_err(|_| "image failed to load".to_string())?;
    Ok(image)
}

fn image_to_bits(image: &HtmlImageElement, model: &Model) -> Result<Inference, String> {
    let width = image.natural_width();
    let height = image.natural_height();
    if width == 0 || height == 0 {
        return Err("image has no pixels".into());
    }

    let document = web_sys::window()
        .ok_or("no window")?
        .document()
        .ok_or("no document")?;
    let canvas = document
        .create_element("canvas")
        .map_err(|_| "could not create canvas")?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| "could not cast canvas")?;
    canvas.set_width(width);
    canvas.set_height(height);
    let context = canvas
        .get_context("2d")
        .map_err(|_| "could not get 2d context")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "could not cast 2d context")?;
    context
        .draw_image_with_html_image_element(image, 0.0, 0.0)
        .map_err(|_| "could not draw image")?;
    let data = context
        .get_image_data(0.0, 0.0, width as f64, height as f64)
        .map_err(|_| "could not read image data")?;
    let rgba = data.data().0;

    let options = PreprocessOptions {
        size: model.config.image_size,
        mode: model.config.binarize,
        threshold: model.config.threshold,
    };
    let bits = rgba_to_bits(width as usize, height as usize, &rgba, options);
    let (key, recon) = model.infer_bits(&bits)?;
    Ok((bits, key, recon))
}
