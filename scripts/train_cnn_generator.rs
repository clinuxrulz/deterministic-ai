//! Tiny conditional CNN decoder for the "regular CNN" comparison tab.
//!
//! This is deliberately *not* part of the workspace: it is an offline weight
//! trainer written against `std` only, so it can be built with plain `rustc`
//! and does not add a dependency or a CI target. The browser reimplements the
//! forward pass in TypeScript; this file is the reproducible source of the
//! weights shipped in `web/public/gan/cnn/model.json`.
//!
//! Architecture (8-bit key -> 32x32 grayscale):
//!
//! ```text
//! key 8 --embedding--> 32 --fc/relu--> 128 --fc/relu--> 4x4x16
//!       --nearest 2x + conv3x3/relu--> 8x8x8
//!       --nearest 2x + conv3x3/relu--> 16x16x4
//!       --nearest 2x + conv3x3/sigmoid--> 32x32x1
//! ```
//!
//! It is a conditional convolutional decoder: each of the 200 Quick Draw cats
//! owns a code (0..199), and the embedding table gives every code its own
//! 32-vector, which is what lets ~46k weights memorise 200 drawings. Codes
//! 200..255 are never trained; they decode to whatever the same weights invent,
//! exactly like the circuit tabs' free codes.
//!
//! Build and run:
//!
//! ```bash
//! rustc -O -C target-cpu=native scripts/train_cnn_generator.rs -o /tmp/train_cnn
//! /tmp/train_cnn data/quickdraw_cats_32_cnn.bin web/public/gan/cnn/model.json 500
//! ```

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const SIDE: usize = 32;
const LATENT: usize = 8;
const CODES: usize = 256;
const EMB: usize = 32;
const HID: usize = 128;
const MAP_H: usize = 4;
const MAP_W: usize = 4;
const MAP_C: usize = 16;
const C1: usize = 8;
const C2: usize = 4;

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        // Box-Muller is overkill; a sum of uniforms is close enough for init.
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.uniform();
        }
        (s - 6.0) / 2.0
    }
}

struct Net {
    embed: Vec<f32>, // CODES * EMB
    w1: Vec<f32>,    // HID * EMB
    b1: Vec<f32>,    // HID
    w2: Vec<f32>,    // (MAP_H*MAP_W*MAP_C) * HID
    b2: Vec<f32>,    // MAP_H*MAP_W*MAP_C
    c1w: Vec<f32>,   // C1 * MAP_C * 3 * 3
    c1b: Vec<f32>,
    c2w: Vec<f32>,   // C2 * C1 * 3 * 3
    c2b: Vec<f32>,
    c3w: Vec<f32>, // 1 * C2 * 3 * 3
    c3b: Vec<f32>,
}

impl Net {
    fn new(rng: &mut Rng) -> Self {
        let mut net = Net {
            embed: vec![0.0; CODES * EMB],
            w1: vec![0.0; HID * EMB],
            b1: vec![0.0; HID],
            w2: vec![0.0; MAP_H * MAP_W * MAP_C * HID],
            b2: vec![0.0; MAP_H * MAP_W * MAP_C],
            c1w: vec![0.0; C1 * MAP_C * 9],
            c1b: vec![0.0; C1],
            c2w: vec![0.0; C2 * C1 * 9],
            c2b: vec![0.0; C2],
            c3w: vec![0.0; C2 * 9],
            c3b: vec![0.0; 1],
        };
        let he = |fan_in: usize, rng: &mut Rng| (2.0 / fan_in as f32).sqrt();
        for x in net.embed.iter_mut() {
            *x = rng.normal() * 0.1;
        }
        let s = he(EMB, rng);
        for x in net.w1.iter_mut() {
            *x = rng.normal() * s;
        }
        let s = he(HID, rng);
        for x in net.w2.iter_mut() {
            *x = rng.normal() * s;
        }
        let s = he(MAP_C * 9, rng);
        for x in net.c1w.iter_mut() {
            *x = rng.normal() * s;
        }
        let s = he(C1 * 9, rng);
        for x in net.c2w.iter_mut() {
            *x = rng.normal() * s;
        }
        let s = he(C2 * 9, rng);
        for x in net.c3w.iter_mut() {
            *x = rng.normal() * s;
        }
        net
    }

    fn n_params(&self) -> usize {
        self.embed.len()
            + self.w1.len()
            + self.b1.len()
            + self.w2.len()
            + self.b2.len()
            + self.c1w.len()
            + self.c1b.len()
            + self.c2w.len()
            + self.c2b.len()
            + self.c3w.len()
            + self.c3b.len()
    }
}

struct Cache {
    code: usize,
    h1: Vec<f32>,
    h2: Vec<f32>,   // MAP_H*MAP_W*MAP_C
    up1: Vec<f32>,  // 8x8xMAP_C
    o1: Vec<f32>,   // 8x8xC1
    up2: Vec<f32>,  // 16x16xC1
    o2: Vec<f32>,   // 16x16xC2
    up3: Vec<f32>,  // 32x32xC2
    out: Vec<f32>,  // 32x32
}

fn forward(net: &Net, code: usize) -> Cache {
    // Embedding + MLP.
    let e = &net.embed[code * EMB..(code + 1) * EMB];
    let mut h1 = vec![0.0; HID];
    for j in 0..HID {
        let mut s = net.b1[j];
        for i in 0..EMB {
            s += e[i] * net.w1[j * EMB + i];
        }
        h1[j] = s.max(0.0);
    }
    let map_n = MAP_H * MAP_W * MAP_C;
    let mut h2 = vec![0.0; map_n];
    for j in 0..map_n {
        let mut s = net.b2[j];
        for i in 0..HID {
            s += h1[i] * net.w2[j * HID + i];
        }
        h2[j] = s.max(0.0);
    }

    let up1 = upsample2(&h2, MAP_H, MAP_W, MAP_C);
    let (o1, _) = conv3(&up1, 2 * MAP_H, 2 * MAP_W, MAP_C, &net.c1w, &net.c1b, C1, true);
    let up2 = upsample2(&o1, 2 * MAP_H, 2 * MAP_W, C1);
    let (o2, _) = conv3(&up2, 4 * MAP_H, 4 * MAP_W, C1, &net.c2w, &net.c2b, C2, true);
    let up3 = upsample2(&o2, 4 * MAP_H, 4 * MAP_W, C2);
    let (out, _) = conv3(&up3, 8 * MAP_H, 8 * MAP_W, C2, &net.c3w, &net.c3b, 1, false);
    let out = out.into_iter().map(sigmoid).collect();
    Cache {
        code,
        h1,
        h2,
        up1,
        o1,
        up2,
        o2,
        up3,
        out,
    }
}

/// Nearest-neighbour 2x upsample, channels-last.
fn upsample2(input: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; (h * 2) * (w * 2) * c];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let v = input[(y * w + x) * c + ch];
                for dy in 0..2 {
                    for dx in 0..2 {
                        out[((y * 2 + dy) * (w * 2) + x * 2 + dx) * c + ch] = v;
                    }
                }
            }
        }
    }
    out
}

fn upsample2_back(grad: &[f32], h: usize, w: usize, c: usize, out: &mut [f32]) {
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let mut s = 0.0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        s += grad[((y * 2 + dy) * (w * 2) + x * 2 + dx) * c + ch];
                    }
                }
                out[(y * w + x) * c + ch] += s;
            }
        }
    }
}

/// 3x3 same convolution with zero padding. Returns output and the pre-activation
/// (unused here, but handy while checking the maths).
fn conv3(
    input: &[f32],
    h: usize,
    w: usize,
    cin: usize,
    weights: &[f32],
    bias: &[f32],
    cout: usize,
    relu: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0; h * w * cout];
    for y in 0..h {
        for x in 0..w {
            for co in 0..cout {
                let mut s = bias[co];
                for ky in 0..3 {
                    let iy = y as isize + ky as isize - 1;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..3 {
                        let ix = x as isize + kx as isize - 1;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        for ci in 0..cin {
                            let wv =
                                weights[((co * cin + ci) * 3 + ky) * 3 + kx];
                            let iv = input[((iy as usize) * w + ix as usize) * cin + ci];
                            s += wv * iv;
                        }
                    }
                }
                out[(y * w + x) * cout + co] = if relu { s.max(0.0) } else { s };
            }
        }
    }
    (out, Vec::new())
}

#[allow(clippy::too_many_arguments)]
fn conv3_back(
    input: &[f32],
    h: usize,
    w: usize,
    cin: usize,
    weights: &[f32],
    out: &[f32],
    cout: usize,
    relu: bool,
    dout: &[f32],
    dinput: &mut [f32],
    dw: &mut [f32],
    db: &mut [f32],
) {
    for y in 0..h {
        for x in 0..w {
            for co in 0..cout {
                let oi = (y * w + x) * cout + co;
                if relu && out[oi] <= 0.0 {
                    continue;
                }
                let dz = dout[oi];
                if dz == 0.0 {
                    continue;
                }
                db[co] += dz;
                for ky in 0..3 {
                    let iy = y as isize + ky as isize - 1;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..3 {
                        let ix = x as isize + kx as isize - 1;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        for ci in 0..cin {
                            let wi = ((co * cin + ci) * 3 + ky) * 3 + kx;
                            let ii = ((iy as usize) * w + ix as usize) * cin + ci;
                            dw[wi] += dz * input[ii];
                            dinput[ii] += dz * weights[wi];
                        }
                    }
                }
            }
        }
    }
}

struct Grads {
    embed: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    c1w: Vec<f32>,
    c1b: Vec<f32>,
    c2w: Vec<f32>,
    c2b: Vec<f32>,
    c3w: Vec<f32>,
    c3b: Vec<f32>,
}

impl Grads {
    fn new(net: &Net) -> Self {
        Grads {
            embed: vec![0.0; net.embed.len()],
            w1: vec![0.0; net.w1.len()],
            b1: vec![0.0; net.b1.len()],
            w2: vec![0.0; net.w2.len()],
            b2: vec![0.0; net.b2.len()],
            c1w: vec![0.0; net.c1w.len()],
            c1b: vec![0.0; net.c1b.len()],
            c2w: vec![0.0; net.c2w.len()],
            c2b: vec![0.0; net.c2b.len()],
            c3w: vec![0.0; net.c3w.len()],
            c3b: vec![0.0; net.c3b.len()],
        }
    }
    fn zero(&mut self) {
        for v in [
            &mut self.embed, &mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2,
            &mut self.c1w, &mut self.c1b, &mut self.c2w, &mut self.c2b,
            &mut self.c3w, &mut self.c3b,
        ] {
            v.fill(0.0);
        }
    }
    fn norm(&self) -> f32 {
        let mut s = 0.0;
        for v in [
            &self.embed, &self.w1, &self.b1, &self.w2, &self.b2,
            &self.c1w, &self.c1b, &self.c2w, &self.c2b,
            &self.c3w, &self.c3b,
        ] {
            for x in v.iter() {
                s += x * x;
            }
        }
        s.sqrt()
    }
}

fn backward(net: &Net, cache: &Cache, target: &[f32], grads: &mut Grads) {
    let n = SIDE * SIDE;
    // BCE-with-logits derivative, averaged over the image.
    let mut dlogits = vec![0.0; n];
    for i in 0..n {
        dlogits[i] = (cache.out[i] - target[i]) / n as f32;
    }

    // conv3 (no relu)
    let h3 = SIDE;
    let w3 = SIDE;
    let mut d_up3 = vec![0.0; h3 * w3 * C2];
    conv3_back(
        &cache.up3, h3, w3, C2, &net.c3w, &cache.out, 1, false,
        &dlogits, &mut d_up3, &mut grads.c3w, &mut grads.c3b,
    );

    // up3 -> d_o2 (16x16xC2)
    let h2 = 16;
    let w2 = 16;
    let mut d_o2 = vec![0.0; h2 * w2 * C2];
    upsample2_back(&d_up3, h2, w2, C2, &mut d_o2);

    // conv2 (relu)
    let mut d_up2 = vec![0.0; h2 * w2 * C1];
    conv3_back(
        &cache.up2, h2, w2, C1, &net.c2w, &cache.o2, C2, true,
        &d_o2, &mut d_up2, &mut grads.c2w, &mut grads.c2b,
    );

    // up2 -> d_o1 (8x8xC1)
    let h1 = 8;
    let w1 = 8;
    let mut d_o1 = vec![0.0; h1 * w1 * C1];
    upsample2_back(&d_up2, h1, w1, C1, &mut d_o1);

    // conv1 (relu)
    let mut d_up1 = vec![0.0; h1 * w1 * MAP_C];
    conv3_back(
        &cache.up1, h1, w1, MAP_C, &net.c1w, &cache.o1, C1, true,
        &d_o1, &mut d_up1, &mut grads.c1w, &mut grads.c1b,
    );

    // up1 -> d_h2 (4x4x16)
    let mut d_h2 = vec![0.0; MAP_H * MAP_W * MAP_C];
    upsample2_back(&d_up1, MAP_H, MAP_W, MAP_C, &mut d_h2);

    // MLP layer 2: h2 = relu(w2*h1+b2)
    let map_n = MAP_H * MAP_W * MAP_C;
    let mut d_h1 = vec![0.0; HID];
    for j in 0..map_n {
        if cache.h2[j] <= 0.0 {
            continue;
        }
        let dz = d_h2[j];
        if dz == 0.0 {
            continue;
        }
        grads.b2[j] += dz;
        for i in 0..HID {
            grads.w2[j * HID + i] += dz * cache.h1[i];
            d_h1[i] += dz * net.w2[j * HID + i];
        }
    }

    // MLP layer 1: h1 = relu(w1*e+b1)
    let e = &net.embed[cache.code * EMB..(cache.code + 1) * EMB];
    for j in 0..HID {
        if cache.h1[j] <= 0.0 {
            continue;
        }
        let dz = d_h1[j];
        if dz == 0.0 {
            continue;
        }
        grads.b1[j] += dz;
        for i in 0..EMB {
            grads.w1[j * EMB + i] += dz * e[i];
            grads.embed[cache.code * EMB + i] += dz * net.w1[j * EMB + i];
        }
    }
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
}

impl Adam {
    fn new(n: usize) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }
    fn step(&mut self, params: &mut [f32], grad: &[f32], lr: f32) {
        self.t += 1;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grad[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let mh = self.m[i] / bc1;
            let vh = self.v[i] / bc2;
            params[i] -= lr * mh / (vh.sqrt() + eps);
        }
    }
}

struct Optim {
    embed: Adam,
    w1: Adam,
    b1: Adam,
    w2: Adam,
    b2: Adam,
    c1w: Adam,
    c1b: Adam,
    c2w: Adam,
    c2b: Adam,
    c3w: Adam,
    c3b: Adam,
}

impl Optim {
    fn new(net: &Net) -> Self {
        Optim {
            embed: Adam::new(net.embed.len()),
            w1: Adam::new(net.w1.len()),
            b1: Adam::new(net.b1.len()),
            w2: Adam::new(net.w2.len()),
            b2: Adam::new(net.b2.len()),
            c1w: Adam::new(net.c1w.len()),
            c1b: Adam::new(net.c1b.len()),
            c2w: Adam::new(net.c2w.len()),
            c2b: Adam::new(net.c2b.len()),
            c3w: Adam::new(net.c3w.len()),
            c3b: Adam::new(net.c3b.len()),
        }
    }
    fn apply(&mut self, net: &mut Net, grads: &Grads, lr: f32) {
        self.embed.step(&mut net.embed, &grads.embed, lr);
        self.w1.step(&mut net.w1, &grads.w1, lr);
        self.b1.step(&mut net.b1, &grads.b1, lr);
        self.w2.step(&mut net.w2, &grads.w2, lr);
        self.b2.step(&mut net.b2, &grads.b2, lr);
        self.c1w.step(&mut net.c1w, &grads.c1w, lr);
        self.c1b.step(&mut net.c1b, &grads.c1b, lr);
        self.c2w.step(&mut net.c2w, &grads.c2w, lr);
        self.c2b.step(&mut net.c2b, &grads.c2b, lr);
        self.c3w.step(&mut net.c3w, &grads.c3w, lr);
        self.c3b.step(&mut net.c3b, &grads.c3b, lr);
    }
}

fn read_data(path: &str) -> (usize, Vec<Vec<u8>>) {
    let buf = fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    assert_eq!(&buf[0..4], b"QCNN", "bad magic in {path}");
    let side = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let px = side * side;
    let mut data = Vec::with_capacity(n);
    let mut off = 12;
    for _ in 0..n {
        data.push(buf[off..off + px].to_vec());
        off += px;
    }
    (side, data)
}

fn write_json(path: &str, net: &Net, trained: usize, epochs: usize, loss: f32) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"kind\": \"cnn\",")?;
    writeln!(f, "  \"side\": {SIDE},")?;
    writeln!(f, "  \"latent\": {LATENT},")?;
    writeln!(f, "  \"trained\": {trained},")?;
    writeln!(f, "  \"params\": {},", net.n_params())?;
    writeln!(f, "  \"epochs\": {epochs},")?;
    writeln!(f, "  \"loss\": {loss:.6},")?;
    writeln!(f, "  \"note\": \"A regular convolutional decoder: 8-bit key -> embedding -> 128 -> 4x4x16, then nearest-neighbour 2x upsampling with 3x3 conv/ReLU at 8x8 and 16x16 and a 3x3 sigmoid head at 32x32. {} weights, trained on the same {} Quick Draw cats as the circuit tabs. Its output is continuous, so it can draw soft edges; the circuit tabs can only return 0/1.\",", net.n_params(), trained)?;
    write_array(&mut f, "embed", &net.embed)?;
    write_array(&mut f, "w1", &net.w1)?;
    write_array(&mut f, "b1", &net.b1)?;
    write_array(&mut f, "w2", &net.w2)?;
    write_array(&mut f, "b2", &net.b2)?;
    write_array(&mut f, "c1w", &net.c1w)?;
    write_array(&mut f, "c1b", &net.c1b)?;
    write_array(&mut f, "c2w", &net.c2w)?;
    write_array(&mut f, "c2b", &net.c2b)?;
    write_array(&mut f, "c3w", &net.c3w)?;
    write_array(&mut f, "c3b", &net.c3b)?;
    writeln!(f, "  \"shapes\": {{\"embed\": [{CODES}, {EMB}], \"w1\": [{HID}, {EMB}], \"b1\": [{HID}], \"w2\": [{} , {HID}], \"b2\": [{}], \"c1w\": [{C1}, {MAP_C}, 3, 3], \"c1b\": [{C1}], \"c2w\": [{C2}, {C1}, 3, 3], \"c2b\": [{C2}], \"c3w\": [1, {C2}, 3, 3], \"c3b\": [1]}}", MAP_H * MAP_W * MAP_C, MAP_H * MAP_W * MAP_C)?;
    writeln!(f, "}}")?;
    Ok(())
}

fn write_array(f: &mut fs::File, name: &str, data: &[f32]) -> std::io::Result<()> {
    write!(f, "  \"{name}\": [")?;
    for (i, v) in data.iter().enumerate() {
        if i > 0 {
            write!(f, ",")?;
        }
        if i % 16 == 0 {
            write!(f, "\n    ")?;
        }
        write!(f, "{v:.6}")?;
    }
    writeln!(f, "\n  ],")?;
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let data_path = args.get(1).map(String::as_str).unwrap_or("data/quickdraw_cats_32_cnn.bin");
    let out_path = args.get(2).map(String::as_str).unwrap_or("web/public/gan/cnn/model.json");
    let epochs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(500);

    let (side, data) = read_data(data_path);
    assert_eq!(side, SIDE, "expected {SIDE}x{SIDE} data, got {side}");
    let n = data.len().min(CODES);
    println!("{n} drawings, {} weights", Net::new(&mut Rng(0x1234_5678_9abc_def0)).n_params());

    let mut rng = Rng(0xdead_beef_1234_5678);
    let mut net = Net::new(&mut rng);
    let mut optim = Optim::new(&net);
    let mut grads = Grads::new(&net);
    let mut order: Vec<usize> = (0..n).collect();

    let batch = 20;
    let lr = 0.003;
    let start = Instant::now();
    let mut last_loss = 0.0;

    for epoch in 0..epochs {
        // Fisher-Yates with the deterministic RNG.
        for i in (1..n).rev() {
            let j = (rng.next() as usize) % (i + 1);
            order.swap(i, j);
        }
        let mut epoch_loss = 0.0;
        for chunk in order.chunks(batch) {
            grads.zero();
            let mut loss = 0.0;
            for &idx in chunk {
                let cache = forward(&net, idx);
                let target: Vec<f32> = data[idx].iter().map(|&b| b as f32).collect();
                for i in 0..target.len() {
                    let p = cache.out[i].clamp(1e-6, 1.0 - 1e-6);
                    loss += -(target[i] * p.ln() + (1.0 - target[i]) * (1.0 - p).ln());
                }
                backward(&net, &cache, &target, &mut grads);
            }
            let scale = 1.0 / chunk.len() as f32;
            for v in [
                &mut grads.embed, &mut grads.w1, &mut grads.b1, &mut grads.w2,
                &mut grads.b2, &mut grads.c1w, &mut grads.c1b, &mut grads.c2w,
                &mut grads.c2b, &mut grads.c3w, &mut grads.c3b,
            ] {
                for x in v.iter_mut() {
                    *x *= scale;
                }
            }
            // Do not move the embedding rows of codes we never trained.
            // Adam's state for them is all zeros, so a zero gradient is a no-op.
            optim.apply(&mut net, &grads, lr);
            epoch_loss += loss / (chunk.len() * SIDE * SIDE) as f32;
        }
        last_loss = epoch_loss / n as f32;
        if epoch % 20 == 0 || epoch + 1 == epochs {
            let secs = start.elapsed().as_secs_f32();
            println!(
                "epoch {:4}  bce {:.4}  grad {:.3}  {:.0}s",
                epoch + 1,
                last_loss,
                grads.norm(),
                secs
            );
        }
    }

    // Measure the final reconstruction error on the trained codes.
    let mut mse = 0.0;
    let mut bce = 0.0;
    for idx in 0..n {
        let cache = forward(&net, idx);
        for i in 0..SIDE * SIDE {
            let t = data[idx][i] as f32;
            let d = cache.out[i] - t;
            mse += d * d;
            let p = cache.out[i].clamp(1e-6, 1.0 - 1e-6);
            bce += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
        }
    }
    mse /= (n * SIDE * SIDE) as f32;
    bce /= (n * SIDE * SIDE) as f32;
    println!("trained-code mse {mse:.5}  bce {bce:.5}");

    write_json(out_path, &net, n, epochs, bce).expect("writing model");
    println!("wrote {out_path}");
}
