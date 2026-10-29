import { createMemo, createSignal, For, onSettled, Repeat, Show } from "solid-js";
import init, { ModelHandle, preprocess_rgba } from "./wasm/pkg/xorae_wasm";
import "./App.css";

const wasmReady = init();

type ModelEntry = {
  id: string;
  name: string;
  path: string;
  description?: string;
};

type ModelIndex = { models: ModelEntry[] };

type SampleEntry = { name: string; path: string };
type SampleIndex = { samples: SampleEntry[] };

type InferenceResult = {
  bits: Uint8Array;
  key: Uint8Array;
  recon: Uint8Array;
};

const base = import.meta.env.BASE_URL;

function asset(path: string): string {
  return `${base}${path.replace(/^\//, "")}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return (await response.json()) as T;
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.src = url;
  await image.decode();
  return image;
}

function imageToBits(image: HTMLImageElement, model: ModelHandle): InferenceResult {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  if (width === 0 || height === 0) throw new Error("image has no pixels");

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("could not get 2d context");

  context.drawImage(image, 0, 0);
  const rgba = new Uint8Array(context.getImageData(0, 0, width, height).data);
  const bits = preprocess_rgba(
    rgba,
    width,
    height,
    model.image_size,
    model.binarize,
    model.threshold,
  );
  const key = model.encode(bits);
  const recon = model.decode_key(key);
  return { bits, key, recon };
}

export default function App() {
  const [models, setModels] = createSignal<ModelEntry[]>([]);
  const [samples, setSamples] = createSignal<SampleEntry[]>([]);
  const [model, setModel] = createSignal<ModelHandle | null>(null);
  const [status, setStatus] = createSignal("loading model index…");
  const [imageUrl, setImageUrl] = createSignal<string | null>(null);
  const [original, setOriginal] = createSignal<Uint8Array>(new Uint8Array());
  const [keyBits, setKeyBits] = createSignal<Uint8Array>(new Uint8Array());
  const [recon, setRecon] = createSignal<Uint8Array>(new Uint8Array());

  const diff = createMemo(() => {
    const a = original();
    const b = recon();
    const n = Math.min(a.length, b.length);
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i += 1) out[i] = a[i] !== b[i] ? 1 : 0;
    return out;
  });

  const ber = createMemo(() => {
    const a = original();
    if (a.length === 0) return 0;
    const b = recon();
    let errors = 0;
    for (let i = 0; i < a.length; i += 1) {
      if (a[i] !== b[i]) errors += 1;
    }
    return errors / a.length;
  });

  const keyHex = createMemo(() => {
    const bits = keyBits();
    if (bits.length === 0) return "";
    let hex = "";
    for (let i = 0; i < bits.length; i += 4) {
      let value = 0;
      for (let j = 0; j < 4 && i + j < bits.length; j += 1) {
        value |= (bits[i + j] & 1) << j;
      }
      hex += value.toString(16);
    }
    return hex;
  });

  function apply(result: InferenceResult, url: string) {
    setOriginal(result.bits);
    setKeyBits(result.key);
    setRecon(result.recon);
    setImageUrl(url);
  }

  async function loadModel(entry: ModelEntry): Promise<ModelHandle> {
    setStatus(`loading ${entry.name}…`);
    const response = await fetch(asset(entry.path));
    if (!response.ok) throw new Error(`${entry.path}: HTTP ${response.status}`);
    const json = await response.text();
    await wasmReady;
    const handle = new ModelHandle(json);
    setModel(handle);
    setStatus(`loaded ${entry.name} (val BER ${(handle.val_ber * 100).toFixed(2)}%)`);
    return handle;
  }

  async function loadSample(sample: SampleEntry, handle?: ModelHandle | null) {
    const current = handle ?? model();
    if (!current) {
      setStatus("Load a model first.");
      return;
    }
    try {
      const url = asset(sample.path);
      setStatus(`running ${sample.name}…`);
      const image = await loadImage(url);
      apply(imageToBits(image, current), url);
      setStatus("done");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function selectModel(index: number) {
    const entry = models()[index];
    if (!entry) return;
    try {
      const handle = await loadModel(entry);
      const first = samples()[0];
      if (first) await loadSample(first, handle);
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function onFile(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const current = model();
    if (!current) {
      setStatus("Load a model first.");
      return;
    }
    const url = URL.createObjectURL(file);
    try {
      const image = await loadImage(url);
      apply(imageToBits(image, current), url);
      setStatus("done");
    } catch (error) {
      setStatus(String(error));
    }
  }

  function toggleKey(index: number) {
    const current = model();
    if (!current) return;
    const next = new Uint8Array(keyBits());
    next[index] ^= 1;
    setKeyBits(next);
    setRecon(current.decode_key(next));
  }

  onSettled(() => {
    void (async () => {
      try {
        await wasmReady;
        const index = await fetchJson<ModelIndex>(asset("models/index.json"));
        if (index.models.length === 0) {
          setStatus("models/index.json is empty");
          return;
        }
        setModels(index.models);
        const handle = await loadModel(index.models[0]);
        try {
          const sampleIndex = await fetchJson<SampleIndex>(asset("samples/index.json"));
          setSamples(sampleIndex.samples);
          if (sampleIndex.samples.length > 0) {
            await loadSample(sampleIndex.samples[0], handle);
          }
        } catch {
          // Samples are optional.
        }
      } catch (error) {
        setStatus(`Could not load models/index.json: ${String(error)}`);
      }
    })();
  });

  return (
    <main class="app">
      <header class="hero">
        <div>
          <p class="eyebrow">deterministic-ai</p>
          <h1>XOR autoencoder</h1>
          <p class="lede">
            A genetic algorithm evolves a wired Boolean circuit. The encoder squishes a{" "}
            {model() ? `${model()!.image_size}×${model()!.image_size}` : "16×16"} image into a
            key; the decoder rebuilds it from the key alone.
          </p>
        </div>
        <div class="model-picker">
          <label for="model">model</label>
          <select
            id="model"
            onChange={(event) => void selectModel(Number(event.currentTarget.value))}
          >
            <For each={models()}>
              {(entry, index) => (
                <option value={index()} title={entry.description}>
                  {entry.name}
                </option>
              )}
            </For>
          </select>
        </div>
      </header>

      <p class="status">{status()}</p>

      <Show when={model()}>
        {(current) => (
          <section class="modelcard">
            <div><span>gate set</span><strong>{current().gate_set}</strong></div>
            <div><span>key</span><strong>{current().key_bits} bits</strong></div>
            <div><span>gates</span><strong>{current().gate_count}</strong></div>
            <div><span>val BER</span><strong>{(current().val_ber * 100).toFixed(2)}%</strong></div>
            <div><span>test BER</span><strong>{(current().test_ber * 100).toFixed(2)}%</strong></div>
            <div><span>seed</span><strong>{current().seed}</strong></div>
          </section>
        )}
      </Show>

      <section class="controls">
        <label class="file-input">
          <span>upload an image</span>
          <input type="file" accept="image/*" onChange={(event) => void onFile(event)} />
        </label>
        <div class="samples">
          <For each={samples()}>
            {(sample) => (
              <button class="sample" onClick={() => void loadSample(sample)}>
                {sample.name}
              </button>
            )}
          </For>
        </div>
      </section>

      <section class="panels">
        <article class="panel">
          <h2>input</h2>
          <div class="preview-wrap">
            <Show when={imageUrl()}>
              {(url) => <img class="preview" src={url()} alt="input" />}
            </Show>
          </div>
          <BitGrid bits={original()} diff={diff()} />
        </article>

        <article class="panel">
          <h2>key</h2>
          <p class="keyhex">{keyHex()}</p>
          <p class="hint">Click a bit to flip it and re-run the decoder.</p>
          <div class="keygrid">
            <Repeat count={keyBits().length}>
              {(index) => (
                <button
                  class={keyBits()[index] === 1 ? "keybit on" : "keybit off"}
                  title={`toggle key bit ${index}`}
                  onClick={() => toggleKey(index)}
                >
                  {index}
                </button>
              )}
            </Repeat>
          </div>
        </article>

        <article class="panel">
          <h2>reconstruction</h2>
          <BitGrid bits={recon()} diff={diff()} />
          <p class="metric">
            bit error rate <strong>{(ber() * 100).toFixed(2)}%</strong>
          </p>
          <p class="hint">Red cells differ from the binarized input.</p>
        </article>
      </section>
    </main>
  );
}

function BitGrid(props: { bits: Uint8Array; diff: Uint8Array }) {
  const side = createMemo(() => Math.max(1, Math.round(Math.sqrt(props.bits.length))));
  return (
    <div
      class="bitgrid"
      style={{ "grid-template-columns": `repeat(${side()}, 1fr)` }}
    >
      <Repeat count={props.bits.length}>
        {(index) => {
          const value = () => props.bits[index];
          const changed = () => props.diff[index] === 1;
          return (
            <div
              class={
                value() === 1
                  ? changed()
                    ? "bit on changed"
                    : "bit on"
                  : changed()
                    ? "bit off changed"
                    : "bit off"
              }
            />
          );
        }}
      </Repeat>
    </div>
  );
}
