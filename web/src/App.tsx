import { createMemo, createSignal, For, onCleanup, onSettled, Repeat, Show } from "solid-js";
import init, { ModelHandle, run_circuit } from "./wasm/pkg/xorae_wasm";
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
  inputRgb: Uint8Array;
  reconRgb: Uint8Array;
};

const base = import.meta.env.BASE_URL;

/** Shared empty overlay: the input panel shows the bits as the circuit sees
 * them, and only the reconstruction marks differences. */
const NO_DIFF = new Uint8Array();

function asset(path: string): string {
  return `${base}${path.replace(/^\//, "")}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return (await response.json()) as T;
}

function bitsToHex(bits: Uint8Array): string {
  let hex = "";
  for (let i = 0; i < bits.length; i += 4) {
    let value = 0;
    for (let j = 0; j < 4 && i + j < bits.length; j += 1) {
      value |= (bits[i + j] & 1) << j;
    }
    hex += value.toString(16);
  }
  return hex;
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.src = url;
  await image.decode();
  return image;
}

function runImage(image: HTMLImageElement, model: ModelHandle): InferenceResult {
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
  const bits = model.preprocess(rgba, width, height);
  const key = model.encode(bits);
  const color = model.color_mode !== "grayscale";
  const recon = color ? new Uint8Array() : model.decode_key(key);
  const inputRgb = color ? model.bits_to_rgb(bits) : new Uint8Array();
  const reconRgb = color ? model.decode_rgb(key) : new Uint8Array();
  return { bits, key, recon, inputRgb, reconRgb };
}

export default function App() {
  const params = new URLSearchParams(window.location.search);
  type Tab = "reconstruct" | "digits" | "gansat" | "gangrad" | "gancnn" | "glyphs" | "sat";
  const initialTab = (value: string | null): Tab => {
    switch (value) {
      case "digits":
        return "digits";
      case "gan":
      case "gansat":
        return "gansat";
      case "gangrad":
        return "gangrad";
      case "cnn":
      case "gancnn":
        return "gancnn";
      case "glyphs":
        return "glyphs";
      case "sat":
        return "sat";
      default:
        return "reconstruct";
    }
  };
  const [tab, setTab] = createSignal<Tab>(initialTab(params.get("tab")));
  const [models, setModels] = createSignal<ModelEntry[]>([]);
  const [selectedId, setSelectedId] = createSignal("");
  const [samples, setSamples] = createSignal<SampleEntry[]>([]);
  const [model, setModel] = createSignal<ModelHandle | null>(null);
  const [status, setStatus] = createSignal("loading model index…");
  const [imageUrl, setImageUrl] = createSignal<string | null>(null);
  const [original, setOriginal] = createSignal<Uint8Array>(new Uint8Array());
  const [keyBits, setKeyBits] = createSignal<Uint8Array>(new Uint8Array());
  const [recon, setRecon] = createSignal<Uint8Array>(new Uint8Array());
  const [inputRgb, setInputRgb] = createSignal<Uint8Array>(new Uint8Array());
  const [reconRgb, setReconRgb] = createSignal<Uint8Array>(new Uint8Array());

  const isColor = () => {
    const current = model();
    return current ? current.color_mode !== "grayscale" : false;
  };

  const diff = createMemo(() => {
    const a = original();
    const b = recon();
    const n = Math.min(a.length, b.length);
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i += 1) out[i] = a[i] !== b[i] ? 1 : 0;
    return out;
  });

  const diffRgb = createMemo(() => {
    const a = inputRgb();
    const b = reconRgb();
    const pixels = Math.floor(Math.min(a.length, b.length) / 3);
    const out = new Uint8Array(pixels);
    for (let p = 0; p < pixels; p += 1) {
      const i = p * 3;
      out[p] = a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2] ? 1 : 0;
    }
    return out;
  });

  const colorMae = createMemo(() => {
    const a = inputRgb();
    const b = reconRgb();
    const n = Math.min(a.length, b.length);
    if (n === 0) return 0;
    let total = 0;
    for (let i = 0; i < n; i += 1) total += Math.abs(a[i] - b[i]);
    return total / (n * 255);
  });

  const pixelAccuracy = createMemo(() => {
    const a = inputRgb();
    const b = reconRgb();
    const pixels = Math.floor(Math.min(a.length, b.length) / 3);
    if (pixels === 0) return 0;
    let correct = 0;
    for (let p = 0; p < pixels; p += 1) {
      const i = p * 3;
      if (a[i] === b[i] && a[i + 1] === b[i + 1] && a[i + 2] === b[i + 2]) correct += 1;
    }
    return correct / pixels;
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

  const keyHex = createMemo(() => (keyBits().length === 0 ? "" : bitsToHex(keyBits())));

  function apply(result: InferenceResult, url: string) {
    setOriginal(result.bits);
    setKeyBits(result.key);
    setRecon(result.recon);
    setInputRgb(result.inputRgb);
    setReconRgb(result.reconRgb);
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
    setSelectedId(entry.id);
    const summary =
      handle.color_mode === "grayscale"
        ? `val BER ${(handle.val_ber * 100).toFixed(2)}%`
        : `abs MAE ${(handle.test_abs_mae * 100).toFixed(2)}%`;
    setStatus(`loaded ${entry.name} (${summary})`);
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
      apply(runImage(image, current), url);
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
      apply(runImage(image, current), url);
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
    if (current.color_mode === "grayscale") {
      setRecon(current.decode_key(next));
    } else {
      setReconRgb(current.decode_rgb(next));
    }
  }

  function rerunKey(next: Uint8Array) {
    const current = model();
    if (!current) return;
    setKeyBits(next);
    if (current.color_mode === "grayscale") {
      setRecon(current.decode_key(next));
    } else {
      setReconRgb(current.decode_rgb(next));
    }
  }

  function flipRandomKey() {
    const bits = keyBits();
    if (bits.length === 0) return;
    const next = new Uint8Array(bits);
    next[Math.floor(Math.random() * bits.length)] ^= 1;
    rerunKey(next);
  }

  function randomiseKey() {
    const bits = keyBits();
    if (bits.length === 0) return;
    const next = new Uint8Array(bits);
    for (let i = 0; i < next.length; i += 1) next[i] = Math.random() < 0.5 ? 1 : 0;
    rerunKey(next);
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
        // ?model=<id> deep-links a model, which is how the headless checks
        // reach one that is not first in the list.
        const wanted = params.get("model");
        const entry = index.models.find((m: ModelEntry) => m.id === wanted) ?? index.models[0];
        const handle = await loadModel(entry);
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
                <option
                  value={index()}
                  selected={entry.id === selectedId()}
                  title={entry.description}
                >
                  {entry.name}
                </option>
              )}
            </For>
          </select>
        </div>
      </header>

      <nav class="tabs" aria-label="mode">
        <button
          class={tab() === "reconstruct" ? "tab active" : "tab"}
          onClick={() => setTab("reconstruct")}
        >
          reconstruct
        </button>
        <button
          class={tab() === "digits" ? "tab active" : "tab"}
          onClick={() => setTab("digits")}
        >
          draw a digit
        </button>
        <button
          class={tab() === "gansat" ? "tab active" : "tab"}
          onClick={() => setTab("gansat")}
        >
          cat gan · sat
        </button>
        <button
          class={tab() === "gangrad" ? "tab active" : "tab"}
          onClick={() => setTab("gangrad")}
        >
          cat gan · grad
        </button>
        <button
          class={tab() === "gancnn" ? "tab active" : "tab"}
          onClick={() => setTab("gancnn")}
        >
          cat gan · cnn
        </button>
        <button
          class={tab() === "glyphs" ? "tab active" : "tab"}
          onClick={() => setTab("glyphs")}
        >
          glyph gan
        </button>
        <button
          class={tab() === "sat" ? "tab active" : "tab"}
          onClick={() => setTab("sat")}
        >
          solver's wiring
        </button>
      </nav>

      <Show
        when={tab() === "reconstruct"}
        fallback={
          <Show
          when={tab() === "digits"}
          fallback={
            <Show
              when={tab() === "gansat"}
              fallback={
                <Show
                  when={tab() === "gangrad"}
                  fallback={
                    <Show
                      when={tab() === "gancnn"}
                      fallback={
                        <Show when={tab() === "glyphs"} fallback={<SatLab />}>
                          <GlyphGan />
                        </Show>
                      }
                    >
                      <CatCnn />
                    </Show>
                  }
                >
                  <CatGan track={CAT_GRAD} />
                </Show>
              }
            >
              <CatGan track={CAT_SAT} />
            </Show>
          }
        >
            <DigitLab />
          </Show>
        }
      >
      <p class="status">{status()}</p>

      <Show when={model()}>
        {(current) => (
          <section class="modelcard">
            <div><span>mode</span><strong>{current().color_mode}</strong></div>
            <div><span>gate set</span><strong>{current().gate_set}</strong></div>
            <div><span>key</span><strong>{current().key_bits} bits</strong></div>
            <div><span>gates</span><strong>{current().gate_count}</strong></div>
            <Show
              when={isColor()}
              fallback={
                <>
                  <div><span>val BER</span><strong>{(current().val_ber * 100).toFixed(2)}%</strong></div>
                  <div><span>test BER</span><strong>{(current().test_ber * 100).toFixed(2)}%</strong></div>
                </>
              }
            >
              <div><span>test abs MAE</span><strong>{(current().test_abs_mae * 100).toFixed(2)}%</strong></div>
              <div><span>test colour MAE</span><strong>{(current().test_color_mae * 100).toFixed(2)}%</strong></div>
              <div><span>test PSNR</span><strong>{current().test_psnr.toFixed(2)} dB</strong></div>
              <div><span>pixel accuracy</span><strong>{(current().test_index_accuracy * 100).toFixed(2)}%</strong></div>
            </Show>
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
          <Show when={isColor()} fallback={<BitGrid bits={original()} diff={NO_DIFF} />}>
            <ColorGrid rgb={inputRgb()} diff={NO_DIFF} />
          </Show>
        </article>

        <article class="panel">
          <h2>key</h2>
          <p class="keyhex">{keyHex()}</p>
          <Show
            when={keyBits().length <= 256}
            fallback={
              <>
                <p class="hint">
                  This key is too long to toggle bit by bit. Flip one at random, or sample a
                  fresh key.
                </p>
                <div class="keyactions">
                  <button class="sample" onClick={flipRandomKey}>
                    flip a random bit
                  </button>
                  <button class="sample" onClick={randomiseKey}>
                    randomise key
                  </button>
                </div>
              </>
            }
          >
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
          </Show>
        </article>

        <article class="panel">
          <h2>reconstruction</h2>
          <Show when={isColor()} fallback={<BitGrid bits={recon()} diff={diff()} />}>
            <ColorGrid rgb={reconRgb()} diff={diffRgb()} />
          </Show>
          <Show
            when={isColor()}
            fallback={
              <p class="metric">
                bit error rate <strong>{(ber() * 100).toFixed(2)}%</strong>
              </p>
            }
          >
            <p class="metric">
              color MAE <strong>{(colorMae() * 100).toFixed(2)}%</strong>
              {" · "}
              pixel accuracy <strong>{(pixelAccuracy() * 100).toFixed(2)}%</strong>
            </p>
          </Show>
          <p class="hint">
            {isColor() ? "Red outlines mark pixels that differ." : "Red cells differ from the binarized input."}
          </p>
        </article>
      </section>
      </Show>
    </main>
  );
}

function ColorGrid(props: { rgb: Uint8Array; diff: Uint8Array }) {
  const side = createMemo(() => Math.max(1, Math.round(Math.sqrt(props.rgb.length / 3))));
  return (
    <div
      class="colorgrid"
      style={{ "grid-template-columns": `repeat(${side()}, 1fr)` }}
    >
      <Repeat count={side() * side()}>
        {(index) => {
          const i = index * 3;
          const color = () => `rgb(${props.rgb[i]}, ${props.rgb[i + 1]}, ${props.rgb[i + 2]})`;
          const changed = () => props.diff[index] === 1;
          return (
            <div
              class={changed() ? "colorcell changed" : "colorcell"}
              style={{ "background-color": color() }}
            />
          );
        }}
      </Repeat>
    </div>
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

// ---------------------------------------------------------------------------
// Draw-a-digit tab: a hand-drawn digit is normalised the way MNIST is
// (bounding box -> 20x20 -> centre of mass -> 28x28 -> 32x32), encoded by the
// MNIST autoencoder, then classified by a NAND circuit over the 128-bit key.
// ---------------------------------------------------------------------------

type DigitMeta = { classes: number; group_size: number; gates: number; test_accuracy: number };
type DigitSample = { label: number; size: number; bits: string };

const DIGIT_CANVAS = 280;
const DIGIT_BG = "#0b0d12";
const DIGIT_INK = "#ffffff";

function toModelBits(source: HTMLCanvasElement): Uint8Array | null {
  const w = source.width;
  const h = source.height;
  const context = source.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  const data = context.getImageData(0, 0, w, h).data;

  let minX = w;
  let minY = h;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = (y * w + x) * 4;
      // White ink on a dark canvas.
      if (data[i + 3] > 32 && data[i] >= 128) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const scale = 20 / Math.max(boxW, boxH);
  const drawW = Math.max(1, Math.round(boxW * scale));
  const drawH = Math.max(1, Math.round(boxH * scale));

  const small = document.createElement("canvas");
  small.width = 28;
  small.height = 28;
  const sctx = small.getContext("2d", { willReadFrequently: true });
  if (!sctx) return null;
  sctx.fillStyle = "#000000";
  sctx.fillRect(0, 0, 28, 28);
  sctx.imageSmoothingEnabled = true;
  sctx.drawImage(
    source,
    minX,
    minY,
    boxW,
    boxH,
    Math.round((28 - drawW) / 2),
    Math.round((28 - drawH) / 2),
    drawW,
    drawH,
  );

  // Centre by centre of mass, as the MNIST preprocessing does.
  const px = sctx.getImageData(0, 0, 28, 28).data;
  let sx = 0;
  let sy = 0;
  let count = 0;
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      if (px[(y * 28 + x) * 4] >= 128) {
        sx += x;
        sy += y;
        count += 1;
      }
    }
  }
  if (count === 0) return null;
  const dx = Math.round(13.5 - sx / count);
  const dy = Math.round(13.5 - sy / count);

  const centred = document.createElement("canvas");
  centred.width = 28;
  centred.height = 28;
  const cctx = centred.getContext("2d", { willReadFrequently: true });
  if (!cctx) return null;
  cctx.fillStyle = "#000000";
  cctx.fillRect(0, 0, 28, 28);
  cctx.drawImage(small, dx, dy);

  // Threshold at 28x28: the classifier reads raw pixel bits, so there is no
  // 28 -> 32 resampling step any more.
  const src = cctx.getImageData(0, 0, 28, 28).data;
  const bits = new Uint8Array(28 * 28);
  for (let i = 0; i < bits.length; i += 1) {
    bits[i] = src[i * 4] >= 128 ? 1 : 0;
  }
  return bits;
}

/**
 * Render recorded stroke points at a fixed MNIST-like width.
 *
 * Drawing a large digit on the canvas and then scaling its *bounding box* to
 * 20 px shrinks the strokes with it, and thin strokes are lost when the
 * bilinear downscale pushes them below the 128 threshold. Re-rendering the
 * path at a constant width makes the result independent of how large the user
 * drew, which is what MNIST looks like.
 */
function strokesToBits(
  strokes: { x: number; y: number }[][],
  width = 2.75,
): Uint8Array | null {
  const points = strokes.flat();
  if (points.length === 0) return null;

  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  const span = Math.max(maxX - minX, maxY - minY);
  if (span <= 1e-4) return null; // a single tap, not a digit
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  // Map the long side of the drawing onto a 20 px box inside the 28 px grid.
  const scale = 20 / span;

  const draw = (dx: number, dy: number): HTMLCanvasElement => {
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("no 2d context");
    context.fillStyle = "#000000";
    context.fillRect(0, 0, 28, 28);
    context.strokeStyle = "#ffffff";
    context.fillStyle = "#ffffff";
    context.lineWidth = width;
    context.lineCap = "round";
    context.lineJoin = "round";
    for (const stroke of strokes) {
      if (stroke.length === 1) {
        const x = 14 + (stroke[0].x - cx) * scale + dx;
        const y = 14 + (stroke[0].y - cy) * scale + dy;
        context.beginPath();
        context.arc(x, y, width / 2, 0, Math.PI * 2);
        context.fill();
        continue;
      }
      context.beginPath();
      stroke.forEach((p, i) => {
        const x = 14 + (p.x - cx) * scale + dx;
        const y = 14 + (p.y - cy) * scale + dy;
        if (i === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
    }
    return canvas;
  };

  // Centre by centre of mass, as MNIST preprocessing does.
  const first = draw(0, 0);
  const data = first.getContext("2d")!.getImageData(0, 0, 28, 28).data;
  let sx = 0;
  let sy = 0;
  let count = 0;
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      if (data[(y * 28 + x) * 4] >= 128) {
        sx += x;
        sy += y;
        count += 1;
      }
    }
  }
  if (count === 0) return null;
  const centred = draw(
    Math.round(13.5 - sx / count),
    Math.round(13.5 - sy / count),
  );
  const src = centred.getContext("2d")!.getImageData(0, 0, 28, 28).data;
  const bits = new Uint8Array(28 * 28);
  for (let i = 0; i < bits.length; i += 1) {
    bits[i] = src[i * 4] >= 128 ? 1 : 0;
  }
  return bits;
}

function DigitLab() {
  const params = new URLSearchParams(window.location.search);
  let drawRef: HTMLCanvasElement | undefined;
  let previewRef: HTMLCanvasElement | undefined;
  let meta: DigitMeta | null = null;
  let circuitJson = "";
  let sample: DigitSample | null = null;
  let drawing = false;
  let lastPoint: { x: number; y: number } | null = null;
  // Live strokes are re-rendered at a fixed width; the built-in sample is a
  // bitmap and goes through the raster path instead.
  let strokes: { x: number; y: number }[][] = [];
  let mode: "strokes" | "image" = "strokes";

  const [status, setStatus] = createSignal("loading digit model…");
  const [prediction, setPrediction] = createSignal<number | null>(null);
  const [votes, setVotes] = createSignal<number[]>([]);
  const [groupSize, setGroupSize] = createSignal(16);
  const [runnerUp, setRunnerUp] = createSignal<{ digit: number; votes: number } | null>(null);

  const clear = () => {
    const canvas = drawRef;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.fillStyle = DIGIT_BG;
    context.fillRect(0, 0, canvas.width, canvas.height);
    strokes = [];
    mode = "strokes";
    setPrediction(null);
    setVotes([]);
    setRunnerUp(null);
  };

  const paintBits = (bits: Uint8Array, side: number) => {
    const canvas = drawRef;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const tile = document.createElement("canvas");
    tile.width = side;
    tile.height = side;
    const tctx = tile.getContext("2d");
    if (!tctx) return;
    const img = tctx.createImageData(side, side);
    for (let i = 0; i < side * side; i += 1) {
      const on = bits[i] === 1;
      const value = on ? 255 : 0;
      img.data[i * 4] = value;
      img.data[i * 4 + 1] = value;
      img.data[i * 4 + 2] = value;
      img.data[i * 4 + 3] = on ? 255 : 0;
    }
    tctx.putImageData(img, 0, 0);
    context.fillStyle = DIGIT_BG;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.imageSmoothingEnabled = false;
    context.drawImage(tile, 0, 0, canvas.width, canvas.height);
  };

  const drawPreview = (bits: Uint8Array) => {
    const canvas = previewRef;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const side = Math.round(Math.sqrt(bits.length));
    const cell = canvas.width / side;
    context.fillStyle = "#05070a";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#e8ecf5";
    for (let y = 0; y < side; y += 1) {
      for (let x = 0; x < side; x += 1) {
        if (bits[y * side + x] === 1) {
          context.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
    }
  };

  const classify = () => {
    const canvas = drawRef;
    if (!canvas || !meta) return;
    const bits = mode === "strokes" ? strokesToBits(strokes) : toModelBits(canvas);
    if (!bits) {
      setPrediction(null);
      setVotes([]);
      return;
    }
    drawPreview(bits);
    const out = run_circuit(circuitJson, bits);
    const classes = meta.classes;
    const group = meta.group_size;
    const scores = Array.from({ length: classes }, (_, c) => {
      let total = 0;
      for (let j = 0; j < group; j += 1) total += out[c * group + j] ?? 0;
      return total;
    });
    let best = 0;
    for (let c = 1; c < classes; c += 1) if (scores[c] > scores[best]) best = c;
    let second = -1;
    for (let c = 0; c < classes; c += 1) {
      if (c !== best && (second < 0 || scores[c] > scores[second])) second = c;
    }
    setGroupSize(group);
    setVotes(scores);
    setPrediction(best);
    setRunnerUp(second >= 0 ? { digit: second, votes: scores[second] } : null);
  };

  // The vote margin is the model's own uncertainty: 16 pins per class, so a
  // one- or two-vote gap means the answer could easily be the runner-up.
  const certaintyText = createMemo(() => {
    const predicted = prediction();
    const runner = runnerUp();
    if (predicted === null || !runner) return "";
    const top = votes()[predicted] ?? 0;
    const close = top - runner.votes <= 2;
    return `${top}/${groupSize()} votes${close ? ` · close call, could be a ${runner.digit}` : ""}`;
  });

  const selfTest = () => {
    if (!sample) return;
    mode = "image";
    const bits = new Uint8Array(sample.size * sample.size);
    for (let i = 0; i < bits.length; i += 1) bits[i] = sample.bits[i] === "1" ? 1 : 0;
    paintBits(bits, sample.size);
    classify();
  };

  const pointOf = (event: PointerEvent): { x: number; y: number } => {
    const canvas = drawRef as HTMLCanvasElement;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const onDown = (event: PointerEvent) => {
    const canvas = drawRef;
    if (!canvas) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    drawing = true;
    mode = "strokes";
    lastPoint = pointOf(event);
    strokes.push([
      { x: lastPoint.x / canvas.width, y: lastPoint.y / canvas.height },
    ]);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.fillStyle = DIGIT_INK;
    context.beginPath();
    context.arc(lastPoint.x, lastPoint.y, 10, 0, Math.PI * 2);
    context.fill();
  };

  const onMove = (event: PointerEvent) => {
    if (!drawing || !lastPoint) return;
    event.preventDefault();
    const canvas = drawRef;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const point = pointOf(event);
    strokes[strokes.length - 1].push({
      x: point.x / canvas.width,
      y: point.y / canvas.height,
    });
    context.strokeStyle = DIGIT_INK;
    context.lineWidth = 20;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(lastPoint.x, lastPoint.y);
    context.lineTo(point.x, point.y);
    context.stroke();
    lastPoint = point;
  };

  const onUp = () => {
    if (!drawing) return;
    drawing = false;
    lastPoint = null;
    classify();
  };

  onSettled(() => {
    void (async () => {
      try {
        const [metaJson, circuitText, sampleJson] = await Promise.all([
          fetchJson<DigitMeta>(asset("digits/meta.json")),
          fetch(asset("digits/classifier.json")).then((r) => {
            if (!r.ok) throw new Error(`classifier.json: HTTP ${r.status}`);
            return r.text();
          }),
          fetchJson<DigitSample>(asset("digits/sample.json")),
        ]);
        await wasmReady;
        meta = metaJson;
        circuitJson = circuitText;
        sample = sampleJson;
        setGroupSize(metaJson.group_size);
        clear();
        setStatus(
          `ready · ${metaJson.gates} NANDINV gates (per-gate NAND/AND choice) · ${(metaJson.test_accuracy * 100).toFixed(1)}% on MNIST test · SAT-proven improvement over plain NAND`,
        );
        // Headless check for the stroke path: ?strokes=<scale> draws a
        // synthetic 7 at that fraction of the canvas and classifies it. The
        // result must not depend on the scale, since width is normalised.
        const strokeScale = Number(params.get("strokes"));
        if (Number.isFinite(strokeScale) && strokeScale > 0) {
          const base = [
            [
              { x: 0.15, y: 0.18 },
              { x: 0.85, y: 0.18 },
            ],
            [
              { x: 0.74, y: 0.18 },
              { x: 0.34, y: 0.88 },
            ],
          ];
          const shrink = (p: { x: number; y: number }) => ({
            x: 0.5 + (p.x - 0.5) * strokeScale,
            y: 0.5 + (p.y - 0.5) * strokeScale,
          });
          strokes = base.map((stroke) => stroke.map(shrink));
          mode = "strokes";
          classify();
          setStatus(`strokes: synthetic 7 at scale ${strokeScale}`);
        }
        if (params.get("selftest") === "1") {
          selfTest();
          setStatus(`self-test: drew a "${sampleJson.label}" and classified it`);
        }
      } catch (error) {
        setStatus(`could not load the digit model: ${String(error)}`);
      }
    })();
  });

  return (
    <section class="digitlab">
      <div class="draw-panel">
        <canvas
          ref={drawRef}
          class="draw-canvas"
          width={DIGIT_CANVAS}
          height={DIGIT_CANVAS}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onPointerLeave={onUp}
        />
        <div class="draw-actions">
          <button class="sample" onClick={clear}>
            clear
          </button>
          <button class="sample" onClick={() => void classify()}>
            read digit
          </button>
          <button class="sample" onClick={selfTest}>
            test sample
          </button>
        </div>
        <p class="status">{status()}</p>
      </div>

      <div class="digit-panel">
        <p class="digit-prediction">{prediction() === null ? "–" : prediction()}</p>
        <p class="hint">predicted digit</p>
        <Show when={certaintyText()}>
          {(text) => <p class="digit-confidence">{text()}</p>}
        </Show>
        <div class="votes">
          <For each={votes()}>
            {(score, index) => (
              <div
                class={
                  index() === prediction()
                    ? "vote winner"
                    : runnerUp()?.digit === index()
                      ? "vote runner"
                      : "vote"
                }
              >
                <span class="vote-label">{index()}</span>
                <div class="vote-bar">
                  <div
                    class="vote-fill"
                    style={{ width: `${(score / Math.max(1, groupSize())) * 100}%` }}
                  />
                </div>
                <span class="vote-count">{score}</span>
              </div>
            )}
          </For>
        </div>
        <canvas ref={previewRef} class="digit-preview" width={128} height={128} />
        <p class="hint">
          what the circuit sees · 28×28 thresholded pixels · stroke width is normalised
        </p>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Cat GAN tab: a generator, not an autoencoder.
//
// `scripts/sat_ae.py --mode fit --gate nandinv --targets data/quickdraw_cats_32.json`
// asks kissat for one wiring of a fixed decoder shape whose code p produces
// Quick Draw cat p. Each gate is NAND or AND (an inversion bit per gate), the
// same gate class as the glyph generator. Codes with a target are fitted to a
// per-pattern slack budget; codes after the last target were never shown a
// drawing and are whatever the same wiring computes.
//
// The information budget is the interesting part: a gate has only a handful of
// bits of wiring freedom, so a 2,000-gate decoder can hold on the order of ten
// 32x32 drawings, not two hundred. The tab samples uniformly from the code
// space and reports how many of the 16 tiles came from fitted codes and how
// far those reconstructions are from their drawings.
// ---------------------------------------------------------------------------

type GanCode = {
  code: string;
  trained: boolean;
  wrong: number | null;
  ink: number;
};

type GanMeta = {
  kind: string;
  side: number;
  latent: number;
  gates: number;
  depth: number;
  trained: number;
  free: number;
  slack: number;
  ber: number;
  exact: number;
  worst: number;
  effective_latent: number;
  source: string;
  codes: GanCode[];
  note: string;
};

const GAN_GRID = 4;
const GAN_GAP = 2; // transparent gutter between tiles, in source pixels
const GAN_ON = 232;
const GAN_OFF = 11;

type CatTrack = {
  dir: string;
  title: string;
  method: "sat" | "grad";
  fittedLabel: string;
  errorLabel: string;
};

type SampleSource = "trained" | "all";

const CAT_SAT: CatTrack = {
  dir: "gan/sat",
  title: "This cat is a circuit.",
  method: "sat",
  fittedLabel: "fitted in grid",
  errorLabel: "fitted error",
};

const CAT_GRAD: CatTrack = {
  dir: "gan/grad",
  title: "This cat is a guess.",
  method: "grad",
  fittedLabel: "known codes",
  errorLabel: "reconstruction error",
};

/** Deterministic RNG for the headless self-test: same grid every load. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function CatGan(props: { track: CatTrack }) {
  const params = new URLSearchParams(window.location.search);
  let canvasRef: HTMLCanvasElement | undefined;
  let circuitJson = "";
  let timer: number | undefined;

  const [info, setInfo] = createSignal<GanMeta | null>(null);
  const [status, setStatus] = createSignal("loading generator circuit…");
  const [auto, setAuto] = createSignal(false);
  // The gradient track only saw a drawing for most, not all, of its codes;
  // the rest are extrapolation, which is what made the tab look cloudy.
  // Default to the codes the trainer actually fitted.
  const [source, setSource] = createSignal<SampleSource>(
    props.track.method === "grad" ? "trained" : "all",
  );
  const [latentHex, setLatentHex] = createSignal("");
  const [density, setDensity] = createSignal(0);
  const [spread, setSpread] = createSignal(0);
  const [fitted, setFitted] = createSignal(0);
  const [fittedError, setFittedError] = createSignal(0);

  /** Little-endian code index, matching how sat_ae.py pins each code. */
  const codeOf = (latent: Uint8Array): number => {
    let code = 0;
    for (let i = 0; i < latent.length; i += 1) if (latent[i]) code |= 1 << i;
    return code;
  };

  const latentOf = (code: string, bits: number): Uint8Array => {
    const latent = new Uint8Array(bits);
    for (let i = 0; i < bits; i += 1) latent[i] = code[i] === "1" ? 1 : 0;
    return latent;
  };

  const generate = (
    meta: GanMeta,
    rng: () => number = Math.random,
    mode: SampleSource = source(),
  ): { density: number; spread: number; fitted: number; error: number } | null => {
    const canvas = canvasRef;
    if (!canvas || circuitJson === "") return null;

    const trained = meta.codes
      .map((entry, index) => ({ entry, index }))
      .filter((item) => item.entry.trained);
    const offset = trained.length > 0 ? Math.floor(rng() * trained.length) : 0;

    const tiles = GAN_GRID * GAN_GRID;
    const samples: Uint8Array[] = [];
    const codes: number[] = [];
    let firstLatent: Uint8Array = new Uint8Array(meta.latent);
    for (let t = 0; t < tiles; t += 1) {
      let latent: Uint8Array;
      if (mode === "trained" && trained.length > 0) {
        const pick = trained[(t + offset) % trained.length];
        latent = latentOf(pick.entry.code, meta.latent);
      } else {
        latent = new Uint8Array(meta.latent);
        for (let i = 0; i < latent.length; i += 1) latent[i] = rng() < 0.5 ? 1 : 0;
      }
      if (t === 0) firstLatent = latent;
      codes.push(codeOf(latent));
      samples.push(run_circuit(circuitJson, latent));
    }

    const side = meta.side;
    const step = side + GAN_GAP;
    const full = step * GAN_GRID - GAN_GAP;
    canvas.width = full;
    canvas.height = full;
    const context = canvas.getContext("2d");
    if (!context) return null;
    const image = context.createImageData(full, full);
    for (let t = 0; t < samples.length; t += 1) {
      const ox = (t % GAN_GRID) * step;
      const oy = Math.floor(t / GAN_GRID) * step;
      const bits = samples[t];
      for (let y = 0; y < side; y += 1) {
        for (let x = 0; x < side; x += 1) {
          const value = bits[y * side + x] === 1 ? GAN_ON : GAN_OFF;
          const p = ((oy + y) * full + ox + x) * 4;
          image.data[p] = value;
          image.data[p + 1] = value;
          image.data[p + 2] = value;
          image.data[p + 3] = 255;
        }
      }
    }
    context.putImageData(image, 0, 0);

    let ones = 0;
    for (const sample of samples) {
      for (const bit of sample) ones += bit;
    }
    const densityValue = ones / (samples.length * samples[0].length);

    // Mean pairwise Hamming: 50% is unrelated, 0% is one image repeated.
    let total = 0;
    let pairs = 0;
    for (let i = 0; i < samples.length; i += 1) {
      for (let j = i + 1; j < samples.length; j += 1) {
        let different = 0;
        for (let k = 0; k < samples[i].length; k += 1) {
          if (samples[i][k] !== samples[j][k]) different += 1;
        }
        total += different / samples[i].length;
        pairs += 1;
      }
    }
    const spreadValue = total / pairs;

    // How many of the 16 sampled codes were shown a drawing, and how far the
    // circuit's output is from that drawing (only fitted codes have a target).
    let fittedCount = 0;
    let wrongBits = 0;
    for (const code of codes) {
      const entry = meta.codes[code];
      if (entry?.trained) {
        fittedCount += 1;
        wrongBits += entry.wrong ?? 0;
      }
    }
    const errorValue = fittedCount > 0 ? wrongBits / (fittedCount * meta.side * meta.side) : 0;

    setLatentHex(bitsToHex(firstLatent));
    setDensity(densityValue);
    setSpread(spreadValue);
    setFitted(fittedCount);
    setFittedError(errorValue);
    return { density: densityValue, spread: spreadValue, fitted: fittedCount, error: errorValue };
  };

  /** Buttons run after the scheduler has flushed, so the signal is current. */
  const reroll = () => {
    const meta = info();
    if (meta) generate(meta);
  };

  const chooseSource = (mode: SampleSource) => {
    setSource(mode);
    const meta = info();
    if (meta) generate(meta, Math.random, mode);
  };

  const toggleAuto = () => {
    if (timer !== undefined) {
      window.clearInterval(timer);
      timer = undefined;
      setAuto(false);
      return;
    }
    reroll();
    timer = window.setInterval(reroll, 1500);
    setAuto(true);
  };

  onCleanup(() => {
    if (timer !== undefined) window.clearInterval(timer);
  });

  onSettled(() => {
    void (async () => {
      try {
        const metaPath = `${props.track.dir}/meta.json`;
        const circuitPath = `${props.track.dir}/circuit.json`;
        const [metaJson, circuitText] = await Promise.all([
          fetchJson<GanMeta>(asset(metaPath)),
          fetch(asset(circuitPath)).then((response) => {
            if (!response.ok) throw new Error(`${circuitPath}: HTTP ${response.status}`);
            return response.text();
          }),
        ]);
        await wasmReady;
        circuitJson = circuitText;
        setInfo(metaJson);
        const gateName = props.track.method === "sat" ? "NAND/AND gates" : "NAND gates";
        setStatus(
          `ready · ${metaJson.gates.toLocaleString()} ${gateName} · ` +
            `${metaJson.latent}-bit key · ${metaJson.trained} fitted of ` +
            `${2 ** metaJson.latent} codes`,
        );
        if (params.get("selftest") === "1") {
          const stats = generate(metaJson, mulberry32(7));
          setStatus(
            `self-test · ${props.track.dir} · ${metaJson.gates} gates · ` +
              `${stats?.fitted ?? 0}/16 fitted · ` +
              `error ${((stats?.error ?? 0) * 100).toFixed(1)}% · ` +
              `density ${((stats?.density ?? 0) * 100).toFixed(1)}% · ` +
              `spread ${((stats?.spread ?? 0) * 100).toFixed(1)}%`,
          );
        } else {
          generate(metaJson);
        }
      } catch (error) {
        setStatus(`could not load the generator: ${String(error)}`);
      }
    })();
  });

  return (
    <section class="ganlab">
      <div class="gan-head">
        <h2>{props.track.title}</h2>
        <p class="hint">
          {info()
            ? `${info()!.trained} Google Quick Draw cats, one ${info()!.latent}-bit key and ` +
              `${info()!.gates.toLocaleString()} literal gates. `
            : ""}
          {props.track.method === "sat"
            ? "The wiring was chosen by kissat, not by gradients: each gate is NAND or AND, and every tile is one key pushed through the same circuit."
            : "The wiring was found by gradient descent through soft NANDs, then snapped back to hard 0/1 gates. Every tile is one key pushed through that decoder; no weights are evaluated."}
        </p>
      </div>

      <div class="gan-panel">
        <canvas ref={canvasRef} class="gancanvas" width={128} height={128} />
        <div class="gan-actions">
          <button class="sample" onClick={reroll}>
            generate 16
          </button>
          <button class={auto() ? "sample on" : "sample"} onClick={toggleAuto}>
            {auto() ? "stop" : "auto-roll"}
          </button>
          <Show when={info() && info()!.trained > 0}>
            <button
              class={source() === "trained" ? "sample on" : "sample"}
              onClick={() => chooseSource("trained")}
            >
              trained codes
            </button>
            <button
              class={source() === "all" ? "sample on" : "sample"}
              onClick={() => chooseSource("all")}
            >
              all codes
            </button>
          </Show>
        </div>
        <p class="status">{status()}</p>
        <p class="hint">
          {info()
            ? props.track.method === "grad"
              ? `Only ${info()!.trained} of the ${2 ** info()!.latent} codes were shown a drawing during training. Trained codes show those drawings; all codes samples the rest of the key space, which is the same wiring extrapolating and is much cloudier.`
              : `Every tile is one key pushed through the solver's wiring. All codes samples the full key space; trained codes shows only the ones a drawing was fitted to.`
            : ""}
        </p>
      </div>

      <div class="gan-side">
        <div class="ganstats">
          <div>
            <span>gates</span>
            <strong>{info() ? info()!.gates.toLocaleString() : "–"}</strong>
          </div>
          <div>
            <span>latent</span>
            <strong>{info() ? `${info()!.latent} bits` : "–"}</strong>
          </div>
          <div>
            <span>depth</span>
            <strong>{info()?.depth ?? "–"}</strong>
          </div>
          <div>
            <span>{props.track.fittedLabel}</span>
            <strong>{`${fitted()}/16`}</strong>
          </div>
          <div>
            <span>{props.track.errorLabel}</span>
            <strong>{(fittedError() * 100).toFixed(1)}% bits</strong>
          </div>
          <div>
            <span>density</span>
            <strong>{(density() * 100).toFixed(1)}%</strong>
          </div>
          <div>
            <span>spread</span>
            <strong>{(spread() * 100).toFixed(1)}%</strong>
          </div>
          <div>
            <span>live key bits</span>
            <strong>{info() ? `${info()!.effective_latent}/${info()!.latent}` : "–"}</strong>
          </div>
        </div>
        <p class="hint">
          first key <code class="ganhex">{latentHex()}</code>
        </p>
        <p class="hint">{info()?.note}</p>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Cat CNN tab: the smooth comparison. This is a regular convolutional decoder,
// not a circuit: an 8-bit key is embedded, expanded to a 4x4x16 map, and
// upsampled twice with 3x3 convolutions. Its output is a probability per pixel,
// so it can draw soft edges -- which is exactly what a literal gate cannot do.
//
// The weights come from scripts/train_cnn_generator.rs (std-only Rust, trained
// offline on the same 200 Quick Draw cats). The browser only runs the forward
// pass; nothing is trained here.
// ---------------------------------------------------------------------------

type CnnModel = {
  kind: string;
  side: number;
  latent: number;
  trained: number;
  params: number;
  epochs: number;
  loss: number;
  note: string;
  embed: number[];
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
  c1w: number[];
  c1b: number[];
  c2w: number[];
  c2b: number[];
  c3w: number[];
  c3b: number[];
};

const CNN_EMB = 32;
const CNN_HID = 128;
const CNN_MAP = 4;
const CNN_MAP_C = 16;
const CNN_C1 = 8;
const CNN_C2 = 4;
const CNN_SCALE = 2; // draw each 32x32 tile at 64x64, then let CSS smooth it

function cnnConv3(
  input: Float32Array,
  h: number,
  w: number,
  cin: number,
  weights: number[],
  bias: number[],
  cout: number,
  relu: boolean,
): Float32Array {
  const out = new Float32Array(h * w * cout);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      for (let co = 0; co < cout; co += 1) {
        let sum = bias[co];
        for (let ky = 0; ky < 3; ky += 1) {
          const iy = y + ky - 1;
          if (iy < 0 || iy >= h) continue;
          for (let kx = 0; kx < 3; kx += 1) {
            const ix = x + kx - 1;
            if (ix < 0 || ix >= w) continue;
            const row = (iy * w + ix) * cin;
            const base = ((co * cin) * 3 + ky) * 3 + kx;
            for (let ci = 0; ci < cin; ci += 1) {
              sum += weights[base + ci * 9] * input[row + ci];
            }
          }
        }
        out[(y * w + x) * cout + co] = relu ? Math.max(0, sum) : sum;
      }
    }
  }
  return out;
}

function cnnUpsample(input: Float32Array, h: number, w: number, c: number): Float32Array {
  const out = new Float32Array(4 * h * w * c);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      for (let ch = 0; ch < c; ch += 1) {
        const value = input[(y * w + x) * c + ch];
        for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            out[((y * 2 + dy) * (w * 2) + x * 2 + dx) * c + ch] = value;
          }
        }
      }
    }
  }
  return out;
}

function cnnForward(model: CnnModel, code: number): Float32Array {
  const e = model.embed.slice(code * CNN_EMB, (code + 1) * CNN_EMB);

  const h1 = new Float32Array(CNN_HID);
  for (let j = 0; j < CNN_HID; j += 1) {
    let sum = model.b1[j];
    const row = j * CNN_EMB;
    for (let i = 0; i < CNN_EMB; i += 1) sum += e[i] * model.w1[row + i];
    h1[j] = Math.max(0, sum);
  }

  const mapN = CNN_MAP * CNN_MAP * CNN_MAP_C;
  const h2 = new Float32Array(mapN);
  for (let j = 0; j < mapN; j += 1) {
    let sum = model.b2[j];
    const row = j * CNN_HID;
    for (let i = 0; i < CNN_HID; i += 1) sum += h1[i] * model.w2[row + i];
    h2[j] = Math.max(0, sum);
  }

  const up1 = cnnUpsample(h2, CNN_MAP, CNN_MAP, CNN_MAP_C);
  const o1 = cnnConv3(up1, 8, 8, CNN_MAP_C, model.c1w, model.c1b, CNN_C1, true);
  const up2 = cnnUpsample(o1, 8, 8, CNN_C1);
  const o2 = cnnConv3(up2, 16, 16, CNN_C1, model.c2w, model.c2b, CNN_C2, true);
  const up3 = cnnUpsample(o2, 16, 16, CNN_C2);
  const o3 = cnnConv3(up3, 32, 32, CNN_C2, model.c3w, model.c3b, 1, false);

  for (let i = 0; i < o3.length; i += 1) {
    const value = o3[i];
    o3[i] = value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
  }
  return o3;
}

function CatCnn() {
  const params = new URLSearchParams(window.location.search);
  let canvasRef: HTMLCanvasElement | undefined;
  let model: CnnModel | null = null;
  let timer: number | undefined;

  const [info, setInfo] = createSignal<CnnModel | null>(null);
  const [status, setStatus] = createSignal("loading CNN weights…");
  const [auto, setAuto] = createSignal(false);
  const [source, setSource] = createSignal<SampleSource>("trained");
  const [codeHex, setCodeHex] = createSignal("");
  const [density, setDensity] = createSignal(0);
  const [spread, setSpread] = createSignal(0);

  const generate = (
    meta: CnnModel,
    rng: () => number = Math.random,
    mode: SampleSource = source(),
  ): { density: number; spread: number } | null => {
    const canvas = canvasRef;
    if (!canvas) return null;

    const tile = meta.side * CNN_SCALE;
    const step = tile + GAN_GAP;
    const full = step * GAN_GRID - GAN_GAP;
    canvas.width = full;
    canvas.height = full;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.clearRect(0, 0, full, full);

    const offscreen = document.createElement("canvas");
    offscreen.width = meta.side;
    offscreen.height = meta.side;
    const octx = offscreen.getContext("2d");
    if (!octx) return null;

    const codes: number[] = [];
    const images: Float32Array[] = [];
    const offset = mode === "trained" ? Math.floor(rng() * Math.max(1, meta.trained)) : 0;
    for (let t = 0; t < GAN_GRID * GAN_GRID; t += 1) {
      const code =
        mode === "trained" && meta.trained > 0
          ? (t + offset) % meta.trained
          : Math.floor(rng() * 2 ** meta.latent);
      const image = cnnForward(meta, code);
      codes.push(code);
      images.push(image);

      const data = octx.createImageData(meta.side, meta.side);
      for (let i = 0; i < image.length; i += 1) {
        const value = Math.round(image[i] * 255);
        data.data[i * 4] = value;
        data.data[i * 4 + 1] = value;
        data.data[i * 4 + 2] = value;
        data.data[i * 4 + 3] = 255;
      }
      octx.putImageData(data, 0, 0);
      const ox = (t % GAN_GRID) * step;
      const oy = Math.floor(t / GAN_GRID) * step;
      context.drawImage(offscreen, ox, oy, tile, tile);
    }

    // Mean probability is ink; mean pairwise L1 is how different the samples are.
    let total = 0;
    for (const image of images) {
      for (const value of image) total += value;
    }
    const densityValue = total / (images.length * images[0].length);
    let difference = 0;
    let pairs = 0;
    for (let i = 0; i < images.length; i += 1) {
      for (let j = i + 1; j < images.length; j += 1) {
        let sum = 0;
        for (let k = 0; k < images[i].length; k += 1) {
          sum += Math.abs(images[i][k] - images[j][k]);
        }
        difference += sum / images[i].length;
        pairs += 1;
      }
    }
    const spreadValue = difference / pairs;

    setCodeHex(codes[0].toString(16).padStart(2, "0"));
    setDensity(densityValue);
    setSpread(spreadValue);
    return { density: densityValue, spread: spreadValue };
  };

  const reroll = () => {
    if (model) generate(model);
  };

  const chooseSource = (mode: SampleSource) => {
    setSource(mode);
    if (model) generate(model, Math.random, mode);
  };

  const toggleAuto = () => {
    if (timer !== undefined) {
      window.clearInterval(timer);
      timer = undefined;
      setAuto(false);
      return;
    }
    reroll();
    timer = window.setInterval(reroll, 1500);
    setAuto(true);
  };

  onCleanup(() => {
    if (timer !== undefined) window.clearInterval(timer);
  });

  onSettled(() => {
    void (async () => {
      try {
        const json = await fetchJson<CnnModel>(asset("gan/cnn/model.json"));
        await wasmReady; // shared init; the forward pass itself is plain JS
        model = json;
        setInfo(json);
        setStatus(
          `ready · ${json.params.toLocaleString()} weights · ${json.latent}-bit key · ` +
            `${json.trained} trained codes · ${json.epochs} epochs`,
        );
        if (params.get("selftest") === "1") {
          const stats = generate(json, mulberry32(7));
          setStatus(
            `self-test · gan/cnn · ${json.params} weights · ` +
              `density ${((stats?.density ?? 0) * 100).toFixed(1)}% · ` +
              `spread ${((stats?.spread ?? 0) * 100).toFixed(1)}%`,
          );
        } else {
          generate(json);
        }
      } catch (error) {
        setStatus(`could not load the CNN: ${String(error)}`);
      }
    })();
  });

  return (
    <section class="ganlab">
      <div class="gan-head">
        <h2>This cat is a CNN.</h2>
        <p class="hint">
          A regular convolutional decoder trained on the same {info()?.trained ?? 200} Quick Draw
          cats: an 8-bit key is embedded, expanded to a 4×4×16 map, then upsampled twice with 3×3
          convolutions. Its output is a probability per pixel, so it can draw soft edges. The
          circuit tabs can only return 0 or 1.
        </p>
      </div>

      <div class="gan-panel">
        <canvas ref={canvasRef} class="gancanvas smooth" width={128} height={128} />
        <div class="gan-actions">
          <button class="sample" onClick={reroll}>
            generate 16
          </button>
          <button class={auto() ? "sample on" : "sample"} onClick={toggleAuto}>
            {auto() ? "stop" : "auto-roll"}
          </button>
          <button
            class={source() === "trained" ? "sample on" : "sample"}
            onClick={() => chooseSource("trained")}
          >
            trained codes
          </button>
          <button
            class={source() === "all" ? "sample on" : "sample"}
            onClick={() => chooseSource("all")}
          >
            all codes
          </button>
        </div>
        <p class="status">{status()}</p>
        <p class="hint">
          Trained codes are the 200 drawings. All codes samples the same 8-bit space the circuit
          tabs sample, including the 56 keys the CNN never saw; those are smooth blends, not
          memorised drawings.
        </p>
      </div>

      <div class="gan-side">
        <div class="ganstats">
          <div>
            <span>weights</span>
            <strong>{info() ? info()!.params.toLocaleString() : "–"}</strong>
          </div>
          <div>
            <span>latent</span>
            <strong>{info() ? `${info()!.latent} bits` : "–"}</strong>
          </div>
          <div>
            <span>trained codes</span>
            <strong>{info()?.trained ?? "–"}</strong>
          </div>
          <div>
            <span>epochs</span>
            <strong>{info()?.epochs ?? "–"}</strong>
          </div>
          <div>
            <span>train BCE</span>
            <strong>{info() ? info()!.loss.toFixed(3) : "–"}</strong>
          </div>
          <div>
            <span>density</span>
            <strong>{(density() * 100).toFixed(1)}%</strong>
          </div>
          <div>
            <span>spread</span>
            <strong>{(spread() * 100).toFixed(1)}%</strong>
          </div>
          <div>
            <span>first code</span>
            <strong>0x{codeHex()}</strong>
          </div>
        </div>
        <p class="hint">
          The circuit tabs' output is a binary image; this one is continuous, which is the whole
          difference between a wire and a weight.
        </p>
        <p class="hint">{info()?.note}</p>
      </div>
    </section>
  );
}


//
// `scripts/sat_ae.py --mode generate --gate nandinv --letters 26 --distance 3`
// asks for a circuit that turns an 8-bit latent into an 8x8 image, where
// "an image" is a hand-written predicate: ink in the alphabet's range, every
// row and column inked, no isolated pixel, every horizontal ink run one of the
// lengths the alphabet itself uses (2, 3, 4, 5 or 8, calibrated from GLYPHS),
// and at least 4 bits from every real glyph.
//
// All 26 letters of the alphabet are pinned to real letters, so the circuit is fitted to
// the entire alphabet; the remaining codes are held to the same rule and come back as glyphs
// that read as letters but are in none of them. That is the whole trick: with
// nothing pinned the solver satisfies the predicate with texture, which was
// checked at 3, 4 and 5 bits. The predicate says "drawn strokes", not "a
// letter", so the alphabet has to be in the circuit for the rest to follow.
//
// It is a GAN only in the loosest sense: the discriminator is that predicate,
// so there is no adversary and nothing is learned.
// ---------------------------------------------------------------------------

type GlyphGanCode = {
  code: string;
  bits: number[];
  name: string | null;
  trained: boolean;
  ink: number;
  nearest_glyph: string;
  distance: number;
};

type GlyphGanMeta = {
  kind: string;
  gate: string;
  shape: string;
  key_bits: number;
  gates: number;
  side: number;
  solver: string;
  seconds: number;
  anchor: number;
  min_distance: number;
  min_ink: number;
  max_ink: number;
  run_lengths: number[];
  codes: GlyphGanCode[];
  predicate: string;
  note: string;
};

function GlyphGan() {
  const params = new URLSearchParams(window.location.search);
  let circuitJson = "";
  let detailRef: HTMLCanvasElement | undefined;

  const [meta, setMeta] = createSignal<GlyphGanMeta | null>(null);
  const [status, setStatus] = createSignal("loading the generator circuit…");
  const [images, setImages] = createSignal<number[][]>([]);
  const [selected, setSelected] = createSignal<number | null>(null);

  /** Decode one code through the shipped circuit, not through the JSON. */
  const decode = (code: string): number[] =>
    Array.from(
      run_circuit(circuitJson, Uint8Array.from([...code].map((c) => (c === "1" ? 1 : 0)))),
    );

  const pick = (index: number, list: number[][], info: GlyphGanMeta | null) => {
    const entry = info?.codes[index];
    const bits = list[index];
    if (!entry || !bits) return;
    setSelected(index);
    drawGlyph(detailRef, bits, info?.side ?? 8);
    setStatus(
      (entry.trained ? `trained letter ${entry.name}` : "new") +
        ` · key ${entry.code} · ${(entry.ink * 100).toFixed(0)}% ink · ` +
        `${entry.distance} bits from ${entry.nearest_glyph}`,
    );
  };

  onSettled(() => {
    void (async () => {
      try {
        const [metaJson, circuit] = await Promise.all([
          fetchJson<GlyphGanMeta>(asset("gan8/meta.json")),
          fetchJson<{ n_inputs: number; gates: unknown[]; outputs: number[] }>(
            asset("gan8/circuit.json"),
          ),
        ]);
        await wasmReady;
        circuitJson = JSON.stringify(circuit);
        setMeta(metaJson);

        // Every code goes through the circuit; the grid shows the circuit's
        // output, and the self-test compares it against the recorded one.
        const decoded = metaJson.codes.map((entry) => decode(entry.code));
        setImages(decoded);

        if (params.get("selftest") === "1") {
          let match = 0;
          for (let i = 0; i < decoded.length; i += 1) {
            const want = metaJson.codes[i].bits;
            const got = decoded[i];
            if (want.length === got.length && want.every((bit, k) => bit === got[k])) match += 1;
          }
          const letters = metaJson.codes.filter((entry) => entry.trained);
          let exact = 0;
          for (let i = 0; i < metaJson.codes.length; i += 1) {
            const entry = metaJson.codes[i];
            if (!entry.trained) continue;
            if (entry.bits.every((bit, k) => bit === decoded[i][k])) exact += 1;
          }
          pick(0, decoded, metaJson);
          setStatus(
            `self-test · ${match}/${decoded.length} codes reproduce · ` +
              `${exact}/${letters.length} letters exact · ${metaJson.gates} gates`,
          );
          return;
        }
        pick(0, decoded, metaJson);
      } catch (error) {
        setStatus(`could not load the generator circuit: ${String(error)}`);
      }
    })();
  });

  const chosen = createMemo(() => {
    const index = selected();
    return index === null ? null : (meta()?.codes[index] ?? null);
  });

  /** The 32 codes are four families of eight: every code's nearest letter. */
  const families = createMemo(() => {
    const info = meta();
    if (!info) return [] as { letter: string; tiles: { entry: GlyphGanCode; index: number }[] }[];
    const groups = new Map<string, { entry: GlyphGanCode; index: number }[]>();
    info.codes.forEach((entry, index) => {
      const list = groups.get(entry.nearest_glyph) ?? [];
      list.push({ entry, index });
      groups.set(entry.nearest_glyph, list);
    });
    return [...groups.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([letter, tiles]) => ({
        letter,
        tiles: tiles.sort((a, b) => a.entry.distance - b.entry.distance),
      }));
  });

  return (
    <section class="ganlab">
      <div class="gan-head">
        <h2>A typeface nothing asked for.</h2>
        <p class="hint">
          The solver was given an {meta()?.key_bits ?? 8}-bit latent and a rule for what a letter looks like, and it chose
          the wiring of {meta()?.gates ?? 3765} gates that satisfies it. All 26 letters are pinned to A through
          Z; the other {meta() ? meta()!.codes.length - 26 : 230} satisfy the same rule and are
          not in the alphabet. Every tile below is drawn by running the circuit.
        </p>
      </div>

      <div class="gan-panel">
        <For each={families()}>
          {(family) => (
            <div class="glyphfamily">
              <span class="glyphfamily-label">{family.letter}</span>
              <div class="glyphgrid">
                <For each={family.tiles}>
                  {(tile) => (
                    <button
                      class={
                        (selected() === tile.index ? "glyphtile on" : "glyphtile") +
                        (tile.entry.trained ? " trained" : "")
                      }
                      onClick={() => {
                        const info = meta();
                        if (info) pick(tile.index, images(), info);
                      }}
                      title={
                        tile.entry.trained
                          ? `the real ${tile.entry.name}`
                          : `new · ${tile.entry.code} · ${tile.entry.distance} bits from ${family.letter}`
                      }
                    >
                      <canvas
                        class="glyphcanvas"
                        ref={(el) =>
                          drawGlyph(
                            el,
                            images()[tile.index] ?? tile.entry.bits,
                            meta()?.side ?? 8,
                          )
                        }
                      />
                    </button>
                  )}
                </For>
              </div>
            </div>
          )}
        </For>
        <p class="hint">
          One row per letter: the first tile is the real letter, followed by variants that satisfy the same
          rule and are in no alphabet.
        </p>
        <p class="status">{status()}</p>
      </div>

      <div class="gan-side">
        <canvas ref={detailRef} class="glyph-detail" width={8} height={8} />
        <p class="hint">
          <Show
            when={chosen()?.trained}
            fallback={
              <>
                not one of the 26 letters: {chosen()?.distance ?? 0} bits from{" "}
                {chosen()?.nearest_glyph ?? "–"}, which is the same order as the alphabet's own
                spacing (C and G are 5 apart, E and F 6).
              </>
            }
          >
            pinned to the real {chosen()?.name}, so the rest of the latent is held to the shape it
            learned here.
          </Show>
        </p>

        <div class="ganstats">
          <div>
            <span>gates</span>
            <strong>{meta() ? meta()!.gates : "–"}</strong>
          </div>
          <div>
            <span>latent</span>
            <strong>{meta() ? `${meta()!.key_bits} bits` : "–"}</strong>
          </div>
          <div>
            <span>codes</span>
            <strong>{meta() ? meta()!.codes.length : "–"}</strong>
          </div>
          <div>
            <span>gate</span>
            <strong>{meta()?.gate ?? "nandinv"}</strong>
          </div>
          <div>
            <span>min distance</span>
            <strong>{meta() ? `${meta()!.min_distance} bits` : "–"}</strong>
          </div>
          <div>
            <span>shape</span>
            <strong>{meta()?.shape ?? "–"}</strong>
          </div>
        </div>

        <p class="hint">
          Why they read as one typeface: each output pixel depends on the latent code bits, where
          the lower bits select the letter body and the remaining bits control variations and stroke
          decorations across the typeface. Measured with <code>scripts/circuit_cones.py</code>.
        </p>
        <p class="hint">{meta()?.predicate}</p>
        <p class="hint">{meta()?.note}</p>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// SAT tab: the wiring was chosen by a solver, not by gradient descent.
//
// `scripts/sat_ae.py --patterns 3 --export web/public/sat` encodes the
// trainer's template (64 -> 32 -> 16 -> 3 -> 16 -> 32 -> 64) as CNF, lets
// Kissat pick both input wires of every gate, and writes the witness as two
// plain `Circuit` JSON objects. The encoder maps 8x8 glyphs to a 3-bit key;
// the decoder maps a key back to a glyph.
//
// The tab is about the *key space*. Three codes are trained, one more is
// reached by the held-out glyphs, and the remaining four are codes the solver
// never had to satisfy, so flipping a bit walks you into a circuit nobody
// asked for. That is the checkpoint's result made clickable: 8 codes, 3
// glyphs, and a decoder that can only draw 16 of the 256 functions of a key.
// ---------------------------------------------------------------------------

type SatGlyph = {
  name: string;
  bits: number[];
  code: string;
  trained: boolean;
  wrong_bits: number;
};

type SatMeta = {
  side: number;
  key_bits: number;
  shape: string;
  gates: number;
  solver: string;
  seconds: number;
  trained: string[];
  trained_codes: string[];
  used_codes: string[];
  glyphs: SatGlyph[];
  gate?: string;
  infeasible: string;
  note: string;
};

type CircuitJson = {
  n_inputs: number;
  gates: { op: string; a: number; b: number }[];
  outputs: number[];
};

type SatCircuits = { encoder: CircuitJson; decoder: CircuitJson };

/** Code string with bit 0 first, which is the order the solver reports. */
function codeBits(value: number, width: number): string {
  let out = "";
  for (let i = 0; i < width; i += 1) out += (value >> i) & 1 ? "1" : "0";
  return out;
}

function drawGlyph(canvas: HTMLCanvasElement | undefined, bits: number[], side: number): void {
  if (!canvas) return;
  canvas.width = side;
  canvas.height = side;
  const context = canvas.getContext("2d");
  if (!context) return;
  const image = context.createImageData(side, side);
  for (let i = 0; i < bits.length && i < side * side; i += 1) {
    const value = bits[i] === 1 ? GAN_ON : GAN_OFF;
    const p = i * 4;
    image.data[p] = value;
    image.data[p + 1] = value;
    image.data[p + 2] = value;
    image.data[p + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

function SatLab() {
  const params = new URLSearchParams(window.location.search);
  let encoderJson = "";
  let decoderJson = "";
  let loaded: SatMeta | null = null;
  let glyphSide = 8;
  let reconRef: HTMLCanvasElement | undefined;

  const [meta, setMeta] = createSignal<SatMeta | null>(null);
  const [status, setStatus] = createSignal("loading the solver's circuit…");
  const [code, setCode] = createSignal("000");
  const [selected, setSelected] = createSignal<string | null>(null);
  const [recon, setRecon] = createSignal<number[]>([]);
  const [space, setSpace] = createSignal<{ code: string; out: number[] }[]>([]);

  const ownersOf = (bits: string, glyphs: SatGlyph[], trained: string[]): string[] =>
    glyphs.filter((glyph) => glyph.code === bits).map((glyph) => glyph.name);

  const describe = (bits: string): string => {
    const owners = ownersOf(bits, loaded?.glyphs ?? [], loaded?.trained ?? []);
    const trained = owners.filter((name) => (loaded?.trained ?? []).includes(name));
    if (trained.length > 0) return `key ${bits} · trained · this is ${trained.join(", ")}`;
    if (owners.length > 0) {
      return `key ${bits} · never trained, but ${owners.join(" and ")} encode here`;
    }
    return `key ${bits} · no glyph lands here — the decoder improvises`;
  };

  /** Redraw from an explicit value: signal reads here would be stale. */
  const show = (next: string, label?: string) => {
    if (decoderJson === "") return;
    const out = Array.from(
      run_circuit(decoderJson, Uint8Array.from([...next].map((c) => (c === "1" ? 1 : 0)))),
    );
    setCode(next);
    setRecon(out);
    drawGlyph(reconRef, out, glyphSide);
    setStatus(label ? `${label} · ${describe(next)}` : describe(next));
  };

  const encodeGlyph = (glyph: SatGlyph) => {
    if (encoderJson === "") return;
    const key = Array.from(run_circuit(encoderJson, Uint8Array.from(glyph.bits)));
    const bits = key.map((bit) => (bit ? "1" : "0")).join("");
    setSelected(glyph.name);
    show(bits, `encode ${glyph.name} → key ${bits}`);
  };

  const mismatch = createMemo(() => {
    const name = selected();
    const out = recon();
    const glyph = (meta()?.glyphs ?? []).find((entry) => entry.name === name);
    if (!glyph || out.length === 0) return null;
    let wrong = 0;
    for (let i = 0; i < out.length; i += 1) if (out[i] !== glyph.bits[i]) wrong += 1;
    return wrong;
  });

  /** Sorted so the chip row reads in the displayed bit order, bit 0 first. */
  const codeList = () => {
    const width = meta()?.key_bits ?? 3;
    return Array.from({ length: 1 << width }, (_, i) => codeBits(i, width)).sort();
  };

  const chipLabel = (bits: string): string => {
    const owners = ownersOf(bits, meta()?.glyphs ?? [], meta()?.trained ?? []);
    if (owners.length === 0) return "—";
    return owners
      .map((name) => ((meta()?.trained ?? []).includes(name) ? name : `${name}·`))
      .join(" ");
  };

  onSettled(() => {
    void (async () => {
      try {
        const [metaJson, circuit] = await Promise.all([
          fetchJson<SatMeta>(asset("sat/meta.json")),
          fetchJson<SatCircuits>(asset("sat/circuit.json")),
        ]);
        await wasmReady;
        encoderJson = JSON.stringify(circuit.encoder);
        decoderJson = JSON.stringify(circuit.decoder);
        loaded = metaJson;
        glyphSide = metaJson.side;
        setMeta(metaJson);

        // Decode every code once: the whole key space is 8 images, so show it.
        setSpace(
          Array.from({ length: 1 << metaJson.key_bits }, (_, i) => {
            const bits = codeBits(i, metaJson.key_bits);
            return {
              code: bits,
              out: Array.from(
                run_circuit(decoderJson, Uint8Array.from([...bits].map((c) => (c === "1" ? 1 : 0)))),
              ),
            };
          }).sort((a, b) => a.code.localeCompare(b.code)),
        );

        const first = metaJson.trained_codes[0] ?? codeBits(0, metaJson.key_bits);

        if (params.get("selftest") === "1") {
          const trained = metaJson.glyphs.filter((glyph) => glyph.trained);
          let exact = 0;
          for (const glyph of trained) {
            const key = run_circuit(encoderJson, Uint8Array.from(glyph.bits));
            const out = run_circuit(decoderJson, key);
            let wrong = 0;
            for (let i = 0; i < out.length; i += 1) if (out[i] !== glyph.bits[i]) wrong += 1;
            if (wrong === 0) exact += 1;
          }
          show(first);
          setStatus(
            `self-test · ${exact}/${trained.length} trained glyphs round-trip exactly · ` +
              `${metaJson.gates} gates · codes ${metaJson.trained_codes.join(" ")}`,
          );
          return;
        }
        show(first);
      } catch (error) {
        setStatus(`could not load the solver's circuit: ${String(error)}`);
      }
    })();
  });

  return (
    <section class="satlab">
      <div class="sat-head">
        <h2>A solver chose these wires.</h2>
        <p class="hint">
          No gradients and no weights: Kissat picked both input wires of each of the{" "}
          {meta()?.gates ?? 163} NAND gates, and it proved the four-glyph prefix cannot
          fit — of all 70 four-glyph subsets exactly one can. Click a letter to run it
          through the encoder, then flip a key bit to decode a code that was never
          trained on.
        </p>
      </div>

      <div class="sat-panel">
        <div class="sat-glyphs">
          <For each={meta()?.glyphs ?? []}>
            {(glyph) => (
              <button
                class={selected() === glyph.name ? "sat-glyph on" : "sat-glyph"}
                onClick={() => encodeGlyph(glyph)}
                title={glyph.trained ? `${glyph.name}: trained` : `${glyph.name}: held out`}
              >
                <canvas class="sattile" ref={(el) => drawGlyph(el, glyph.bits, meta()?.side ?? 8)} />
                <span class="sattile-label">
                  <strong>{glyph.name}</strong>
                  <em class={glyph.trained ? "sat-trained" : undefined}>
                    {glyph.trained ? `${glyph.code} · trained` : glyph.code}
                  </em>
                </span>
              </button>
            )}
          </For>
        </div>
        <p class="hint">
          The code under each letter is the encoder's answer. Three glyphs were trained, one
          per code; the five held-out ones land on those same three and on one more code
          that nothing was trained on.
        </p>
        <span class="sat-keyhead">the whole key space — 8 codes, 8 images</span>
        <div class="sat-space">
          <For each={space()}>
            {(entry) => (
              <button
                class={entry.code === code() ? "sat-cell on" : "sat-cell"}
                onClick={() => {
                  setSelected(null);
                  show(entry.code);
                }}
                title={describe(entry.code)}
              >
                <canvas class="sattile" ref={(el) => drawGlyph(el, entry.out, meta()?.side ?? 8)} />
                <em>{entry.code}</em>
              </button>
            )}
          </For>
        </div>
        <p class="hint">
          Every code the decoder can be handed. Five of these were never trained on, yet
          each still produces a definite image — that is the code space doing the work, not
          memorised pictures.
        </p>
      </div>

      <div class="sat-side">
        <div class="sat-key">
          <span class="sat-keyhead">key</span>
          <For each={Array.from({ length: meta()?.key_bits ?? 3 }, (_, i) => i)}>
            {(bit) => (
              <button
                class="sat-bit"
                onClick={() => {
                  setSelected(null);
                  const next = [...code()];
                  next[bit] = next[bit] === "1" ? "0" : "1";
                  show(next.join(""));
                }}
                title={`flip key bit ${bit}`}
              >
                {code()[bit]}
                <em>k{bit}</em>
              </button>
            )}
          </For>
        </div>

        <div class="sat-codes">
          <For each={codeList()}>
            {(bits) => (
              <button
                class={bits === code() ? "sat-chip on" : "sat-chip"}
                onClick={() => {
                  setSelected(null);
                  show(bits);
                }}
              >
                {bits}
                <em>{chipLabel(bits)}</em>
              </button>
            )}
          </For>
        </div>

        <p class="hint sat-chipnote">
          A chip's label lists the glyphs that encode to that code; <code>·</code> marks a
          glyph the solver never had to fit, and <code>—</code> a code no glyph reaches.
        </p>

        <span class="sat-keyhead">decoder output · key {code()}</span>
        <canvas ref={reconRef} class="sat-recon" width={64} height={64} />
        <p class="status">{status()}</p>
        <Show when={mismatch() !== null}>
          <p class="hint">
            {mismatch()} of {recon().length} bits differ from {selected()} — this key was
            not trained on it.
          </p>
        </Show>

        <div class="ganstats">
          <div>
            <span>gates</span>
            <strong>{meta() ? meta()!.gates : "–"}</strong>
          </div>
          <div>
            <span>key</span>
            <strong>{meta() ? `${meta()!.key_bits} bits` : "–"}</strong>
          </div>
          <div>
            <span>shape</span>
            <strong>{meta()?.shape ?? "–"}</strong>
          </div>
          <div>
            <span>solver</span>
            <strong>{meta()?.solver ?? "–"}</strong>
          </div>
        </div>
        <p class="hint">{meta()?.infeasible}</p>
      </div>

      <SatInvert />
    </section>
  );
}

// ---------------------------------------------------------------------------
// The same shape, one bit richer per gate: NAND or AND. Same 163 gates, but a
// gate may invert its own output, so a complemented wire does not have to be
// paid for in the next layer. Six glyphs fit exactly instead of three.
// ---------------------------------------------------------------------------

function SatInvert() {
  const params = new URLSearchParams(window.location.search);
  let encoderJson = "";
  let decoderJson = "";
  let loaded: SatMeta | null = null;
  let reconRef: HTMLCanvasElement | undefined;

  const [meta, setMeta] = createSignal<SatMeta | null>(null);
  const [status, setStatus] = createSignal("loading the second circuit…");
  const [code, setCode] = createSignal("000");
  const [selected, setSelected] = createSignal<string | null>(null);
  const [recon, setRecon] = createSignal<number[]>([]);

  /** Redraw from an explicit value: signal reads here would be stale. */
  const show = (next: string, label?: string) => {
    if (decoderJson === "") return;
    const out = Array.from(
      run_circuit(decoderJson, Uint8Array.from([...next].map((c) => (c === "1" ? 1 : 0)))),
    );
    setCode(next);
    setRecon(out);
    drawGlyph(reconRef, out, loaded?.side ?? 8);
    setStatus(label ?? `key ${next} · decoded`);
  };

  const encodeGlyph = (glyph: SatGlyph) => {
    if (encoderJson === "") return;
    const key = Array.from(run_circuit(encoderJson, Uint8Array.from(glyph.bits)));
    const bits = key.map((bit) => (bit ? "1" : "0")).join("");
    setSelected(glyph.name);
    const note = glyph.trained ? "trained, exact" : `${glyph.wrong_bits} bits wrong`;
    show(bits, `encode ${glyph.name} → key ${bits} · ${note}`);
  };

  const heldOut = createMemo(() => (meta()?.glyphs ?? []).filter((glyph) => !glyph.trained));

  const mismatch = createMemo(() => {
    const name = selected();
    const out = recon();
    const glyph = (meta()?.glyphs ?? []).find((entry) => entry.name === name);
    if (!glyph || out.length === 0) return null;
    let wrong = 0;
    for (let i = 0; i < out.length; i += 1) if (out[i] !== glyph.bits[i]) wrong += 1;
    return wrong;
  });

  onSettled(() => {
    void (async () => {
      try {
        const [metaJson, circuit] = await Promise.all([
          fetchJson<SatMeta>(asset("sat/nandinv/meta.json")),
          fetchJson<SatCircuits>(asset("sat/nandinv/circuit.json")),
        ]);
        await wasmReady;
        encoderJson = JSON.stringify(circuit.encoder);
        decoderJson = JSON.stringify(circuit.decoder);
        loaded = metaJson;
        setMeta(metaJson);

        const first = metaJson.trained_codes[0] ?? codeBits(0, metaJson.key_bits);

        if (params.get("selftest") === "1") {
          const trained = metaJson.glyphs.filter((glyph) => glyph.trained);
          let exact = 0;
          for (const glyph of trained) {
            const key = run_circuit(encoderJson, Uint8Array.from(glyph.bits));
            const out = run_circuit(decoderJson, key);
            let wrong = 0;
            for (let i = 0; i < out.length; i += 1) if (out[i] !== glyph.bits[i]) wrong += 1;
            if (wrong === 0) exact += 1;
          }
          show(first);
          setStatus(
            `self-test · ${exact}/${trained.length} glyphs round-trip exactly · ` +
              `${metaJson.gates} gates · keys ${metaJson.trained_codes.join(" ")}`,
          );
          return;
        }
        show(first);
      } catch (error) {
        setStatus(`could not load the inverting circuit: ${String(error)}`);
      }
    })();
  });

  return (
    <div class="sat-inv">
      <div class="sat-head">
        <h2>One more bit per gate, and six glyphs fit.</h2>
        <p class="hint">
          Same shape, same {meta()?.gates ?? 163} gates, but each gate may also invert its
          own output — NAND or AND. A gate can then hand a complemented wire to its own
          layer instead of spending a whole gate in the next one, so the decoder reaches 58
          of the 256 functions of a 3-bit key instead of 16. Six of the eight glyphs now
          fit exactly, and all 70 four-glyph subsets fit where NAND fits one of the 70.
        </p>
      </div>

      <div class="sat-panel">
        <div class="sat-glyphs">
          <For each={meta()?.glyphs ?? []}>
            {(glyph) => (
              <button
                class={selected() === glyph.name ? "sat-glyph on" : "sat-glyph"}
                onClick={() => encodeGlyph(glyph)}
                title={glyph.trained ? `${glyph.name}: trained` : `${glyph.name}: held out`}
              >
                <canvas class="sattile" ref={(el) => drawGlyph(el, glyph.bits, meta()?.side ?? 8)} />
                <span class="sattile-label">
                  <strong>{glyph.name}</strong>
                  <em class={glyph.trained ? "sat-trained" : undefined}>
                    {glyph.trained ? `${glyph.code} · trained` : glyph.code}
                  </em>
                </span>
              </button>
            )}
          </For>
        </div>
        <p class="hint">
          Six glyphs were trained, one per code. The {heldOut().length} held-out glyphs
          land on codes that were trained on, so all eight images decode from six codes —
          the code space is doing the work, not a lookup table of eight.
        </p>
      </div>

      <div class="sat-side">
        <div class="sat-key">
          <span class="sat-keyhead">key</span>
          <For each={Array.from({ length: meta()?.key_bits ?? 3 }, (_, i) => i)}>
            {(bit) => (
              <button
                class="sat-bit"
                onClick={() => {
                  setSelected(null);
                  const next = [...code()];
                  next[bit] = next[bit] === "1" ? "0" : "1";
                  show(next.join(""));
                }}
                title={`flip key bit ${bit}`}
              >
                {code()[bit]}
                <em>k{bit}</em>
              </button>
            )}
          </For>
        </div>

        <span class="sat-keyhead">decoder output · key {code()}</span>
        <canvas ref={reconRef} class="sat-recon" width={64} height={64} />
        <p class="status">{status()}</p>
        <Show when={mismatch() !== null && selected() !== null}>
          <p class="hint">
            {mismatch()} of {recon().length} bits differ from {selected()}.
          </p>
        </Show>

        <div class="ganstats">
          <div>
            <span>gates</span>
            <strong>{meta() ? meta()!.gates : "–"}</strong>
          </div>
          <div>
            <span>key</span>
            <strong>{meta() ? `${meta()!.key_bits} bits` : "–"}</strong>
          </div>
          <div>
            <span>gate</span>
            <strong>{meta()?.gate ?? "nandinv"}</strong>
          </div>
          <div>
            <span>shape</span>
            <strong>{meta()?.shape ?? "–"}</strong>
          </div>
        </div>
        <p class="hint">{meta()?.note}</p>
      </div>
    </div>
  );
}
