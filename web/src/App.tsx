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
  const [tab, setTab] = createSignal<"reconstruct" | "digits" | "gan">(
    params.get("tab") === "digits"
      ? "digits"
      : params.get("tab") === "gan"
        ? "gan"
        : "reconstruct",
  );
  const [models, setModels] = createSignal<ModelEntry[]>([]);
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
          class={tab() === "gan" ? "tab active" : "tab"}
          onClick={() => setTab("gan")}
        >
          cat gan
        </button>
      </nav>

      <Show
        when={tab() === "reconstruct"}
        fallback={
          <Show when={tab() === "digits"} fallback={<CatGan />}>
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
              <div><span>test palette MAE</span><strong>{(current().test_color_mae * 100).toFixed(2)}%</strong></div>
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
          <Show when={isColor()} fallback={<BitGrid bits={original()} diff={diff()} />}>
            <ColorGrid rgb={inputRgb()} diff={diffRgb()} />
          </Show>
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
          `ready · ${metaJson.gates} NAND gates, no weights · ${(metaJson.test_accuracy * 100).toFixed(1)}% on MNIST test`,
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
// Cat GAN tab: the generator is a wired NAND circuit, not a CNN with weights.
// A random 96-bit latent is pushed through 2,304 gates and the 1,024 outputs
// are drawn as a 32x32 image.
//
// It is a placeholder. The circuit reproduces cat-image statistics (density,
// local texture) but not anatomy, and only about 15 of its 96 latent bits
// change the output, so the grid shows large smooth blobs rather than cats.
// `effective_latent` in gan/meta.json is that number, measured offline by
// flipping each latent bit and counting changed outputs.
// ---------------------------------------------------------------------------

type GanMeta = {
  side: number;
  latent: number;
  gates: number;
  depth: number;
  active_gates: number;
  effective_latent: number;
  dead_latent: number;
  note: string;
};

const GAN_GRID = 4;
const GAN_GAP = 2; // transparent gutter between tiles, in source pixels
const GAN_ON = 232;
const GAN_OFF = 11;

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

function CatGan() {
  const params = new URLSearchParams(window.location.search);
  let canvasRef: HTMLCanvasElement | undefined;
  let circuitJson = "";
  let timer: number | undefined;

  const [info, setInfo] = createSignal<GanMeta | null>(null);
  const [status, setStatus] = createSignal("loading generator circuit…");
  const [auto, setAuto] = createSignal(false);
  const [latentHex, setLatentHex] = createSignal("");
  const [density, setDensity] = createSignal(0);
  const [spread, setSpread] = createSignal(0);

  const generate = (
    meta: GanMeta,
    rng: () => number = Math.random,
  ): { density: number; spread: number } | null => {
    const canvas = canvasRef;
    if (!canvas || circuitJson === "") return null;

    const tiles = GAN_GRID * GAN_GRID;
    const samples: Uint8Array[] = [];
    let firstLatent = new Uint8Array(meta.latent);
    for (let t = 0; t < tiles; t += 1) {
      const latent = new Uint8Array(meta.latent);
      for (let i = 0; i < latent.length; i += 1) latent[i] = rng() < 0.5 ? 1 : 0;
      if (t === 0) firstLatent = latent;
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

    setLatentHex(bitsToHex(firstLatent));
    setDensity(densityValue);
    setSpread(spreadValue);
    return { density: densityValue, spread: spreadValue };
  };

  /** Buttons run after the scheduler has flushed, so the signal is current. */
  const reroll = () => {
    const meta = info();
    if (meta) generate(meta);
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
        const [metaJson, circuitText] = await Promise.all([
          fetchJson<GanMeta>(asset("gan/meta.json")),
          fetch(asset("gan/circuit.json")).then((response) => {
            if (!response.ok) throw new Error(`gan/circuit.json: HTTP ${response.status}`);
            return response.text();
          }),
        ]);
        await wasmReady;
        circuitJson = circuitText;
        setInfo(metaJson);
        setStatus(
          `ready · ${metaJson.gates.toLocaleString()} NAND gates · ` +
            `${metaJson.latent}-bit latent · depth ${metaJson.depth}`,
        );
        if (params.get("selftest") === "1") {
          const stats = generate(metaJson, mulberry32(7));
          setStatus(
            `self-test · ${metaJson.gates} gates · ` +
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
        <h2>This cat is not real.</h2>
        <p class="hint">
          2,304 literal NAND gates and 96 random bits. No weights are evaluated to make these images
          — each tile is one latent, sampled fresh.
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
        </div>
        <p class="status">{status()}</p>
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
            <span>density</span>
            <strong>{(density() * 100).toFixed(1)}%</strong>
          </div>
          <div>
            <span>spread</span>
            <strong>{(spread() * 100).toFixed(1)}%</strong>
          </div>
          <div>
            <span>live latent bits</span>
            <strong>{info() ? `${info()!.effective_latent}/${info()!.latent}` : "–"}</strong>
          </div>
        </div>
        <p class="hint">
          latent <code class="ganhex">{latentHex()}</code>
        </p>
        <p class="hint">{info()?.note}</p>
      </div>
    </section>
  );
}
