//! Tolerance-aware scoring for line-drawing (edge) targets.
//!
//! A per-pixel error treats a stroke that is one pixel off as a total miss,
//! which is exactly how a coarse circuit misses: on a 32x32 edge map with
//! ~150 ink pixels, a perfect drawing shifted by one pixel scores F1 0.0 and
//! BER 0.30, identically to noise. Edge-detection benchmarks instead ask
//! whether a predicted pixel is *near* the target's ink and vice versa.
//!
//! Two numbers come out of it:
//!
//! - `F1@r` -- precision and recall with a radius of `r` pixels, the standard
//!   tolerant boundary score;
//! - `chamfer` -- the mean distance, both directions, in pixels. That is the
//!   number to watch: it degrades smoothly instead of falling off a cliff.
//!
//! Precision and recall are both needed. Recall alone is satisfied by filling
//! the image with ink; precision alone by drawing nothing.

use xorae_core::bits::BitBatch;

/// Radius 1.5 counts a diagonal neighbour (1.414) as hit, which is what "off
/// by one pixel" means in a pixel grid.
pub const RADIUS_NEAR: f32 = 1.5;
/// Radius 3 allows a two-pixel slip; a much looser test, reported separately.
pub const RADIUS_FAR: f32 = 3.0;

/// Two-pass (1, sqrt2) chamfer distance to the nearest ink pixel, in pixels.
/// Zero where the pixel itself is ink.
pub fn distance_to_ink(bits: &[u8], side: usize) -> Vec<f32> {
    const INF: f32 = 1e9;
    let (d1, d2) = (1.0f32, std::f32::consts::SQRT_2);
    let mut d = vec![INF; bits.len()];
    for (i, &b) in bits.iter().enumerate() {
        if b == 1 {
            d[i] = 0.0;
        }
    }
    for y in 0..side {
        for x in 0..side {
            let i = y * side + x;
            let mut best = d[i];
            if y > 0 {
                best = best.min(d[i - side] + d1);
                if x > 0 {
                    best = best.min(d[i - side - 1] + d2);
                }
                if x + 1 < side {
                    best = best.min(d[i - side + 1] + d2);
                }
            }
            if x > 0 {
                best = best.min(d[i - 1] + d1);
            }
            d[i] = best;
        }
    }
    for y in (0..side).rev() {
        for x in (0..side).rev() {
            let i = y * side + x;
            let mut best = d[i];
            if y + 1 < side {
                best = best.min(d[i + side] + d1);
                if x > 0 {
                    best = best.min(d[i + side - 1] + d2);
                }
                if x + 1 < side {
                    best = best.min(d[i + side + 1] + d2);
                }
            }
            if x + 1 < side {
                best = best.min(d[i + 1] + d1);
            }
            d[i] = best;
        }
    }
    // A blank image leaves every entry at the sentinel. That is not a distance:
    // clamp it to the farthest apart two pixels can be (the diagonal), so an
    // empty prediction scores "as wrong as possible" instead of poisoning the
    // mean with 1e9. Without this a single all-zero reconstruction made the
    // reported chamfer 49,000,008 px.
    let cap = side as f32 * std::f32::consts::SQRT_2;
    for v in d.iter_mut() {
        if *v >= INF {
            *v = cap;
        }
    }
    d
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeScores {
    pub f1_near: f32,
    pub f1_far: f32,
    /// Mean distance from ink to the nearest opposite pixels, in pixels.
    pub chamfer: f32,
    pub target_density: f32,
    pub pred_density: f32,
    /// The same F1/chamfer for pseudo-random ink at the same density. At a
    /// 1.5 px radius on a 15%-ink 32x32 target this is about 0.56, so a bare
    /// 0.57 means "no better than spraying" -- the score is only readable
    /// against its own chance level.
    pub chance_f1_near: f32,
    pub chance_chamfer: f32,
}

/// Score one reconstruction against its target, both `side * side` bits.
pub fn scores(target: &[u8], pred: &[u8], side: usize) -> EdgeScores {
    let real = raw(target, pred, side);
    // Chance level: the same amount of ink, placed at random. A deterministic
    // xorshift keeps the number reproducible across runs.
    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ (real.pred_density * 1e6) as u64;
    let mut shuffled = vec![0u8; side * side];
    let cut = (real.pred_density * 1000.0) as u64;
    for slot in shuffled.iter_mut() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        if rng % 1000 < cut {
            *slot = 1;
        }
    }
    let chance = raw(target, &shuffled, side);
    EdgeScores {
        chance_f1_near: chance.f1_near,
        chance_chamfer: chance.chamfer,
        ..real
    }
}

fn raw(target: &[u8], pred: &[u8], side: usize) -> EdgeScores {
    assert_eq!(target.len(), side * side);
    assert_eq!(pred.len(), side * side);
    let n = side * side;
    let to_target = distance_to_ink(target, side);
    let to_pred = distance_to_ink(pred, side);

    let mut ink_target = 0usize;
    let mut ink_pred = 0usize;
    let (mut hit_near, mut hit_far) = (0usize, 0usize); // predicted near target
    let (mut found_near, mut found_far) = (0usize, 0usize); // target near predicted
    let (mut dist_pred, mut dist_target) = (0f64, 0f64);

    for i in 0..n {
        if pred[i] == 1 {
            ink_pred += 1;
            dist_pred += to_target[i] as f64;
            if to_target[i] <= RADIUS_NEAR + 1e-3 {
                hit_near += 1;
            }
            if to_target[i] <= RADIUS_FAR + 1e-3 {
                hit_far += 1;
            }
        }
        if target[i] == 1 {
            ink_target += 1;
            dist_target += to_pred[i] as f64;
            if to_pred[i] <= RADIUS_NEAR + 1e-3 {
                found_near += 1;
            }
            if to_pred[i] <= RADIUS_FAR + 1e-3 {
                found_far += 1;
            }
        }
    }

    let density = |k: usize| k as f32 / n as f32;
    // An empty target and an empty prediction is a perfect match; an empty
    // prediction against real ink is a total miss, and vice versa.
    if ink_target == 0 && ink_pred == 0 {
        return EdgeScores {
            f1_near: 1.0,
            f1_far: 1.0,
            chamfer: 0.0,
            target_density: 0.0,
            pred_density: 0.0,
            ..Default::default()
        };
    }
    let f1 = |precision: f32, recall: f32| {
        if precision + recall <= 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    };
    let precision = |hits: usize| hits as f32 / ink_pred.max(1) as f32;
    let recall = |found: usize| found as f32 / ink_target.max(1) as f32;
    let denom = (ink_pred + ink_target).max(1) as f64;
    EdgeScores {
        f1_near: f1(precision(hit_near), recall(found_near)),
        f1_far: f1(precision(hit_far), recall(found_far)),
        chamfer: ((dist_pred + dist_target) / denom) as f32,
        target_density: density(ink_target),
        pred_density: density(ink_pred),
        ..Default::default()
    }
}

/// Average the scores over a whole evaluation batch. Returns `None` when the
/// columns are not a square image (palette or colour batches), where a pixel
/// distance is not meaningful.
pub fn batch_scores(recon: &[u64], batch: &BitBatch) -> Option<EdgeScores> {
    let side = (batch.cols as f64).sqrt().round() as usize;
    if batch.len == 0 || side * side != batch.cols {
        return None;
    }
    let mut total = EdgeScores::default();
    let mut target = vec![0u8; batch.cols];
    let mut pred = vec![0u8; batch.cols];
    for sample in 0..batch.len {
        let word = sample / 64;
        let bit = sample % 64;
        for col in 0..batch.cols {
            target[col] = ((batch.planes[col * batch.words + word] >> bit) & 1) as u8;
            pred[col] = ((recon[col * batch.words + word] >> bit) & 1) as u8;
        }
        let s = scores(&target, &pred, side);
        total.f1_near += s.f1_near;
        total.f1_far += s.f1_far;
        total.chamfer += s.chamfer;
        total.target_density += s.target_density;
        total.pred_density += s.pred_density;
        total.chance_f1_near += s.chance_f1_near;
        total.chance_chamfer += s.chance_chamfer;
    }
    let n = batch.len as f32;
    Some(EdgeScores {
        f1_near: total.f1_near / n,
        f1_far: total.f1_far / n,
        chamfer: total.chamfer / n,
        target_density: total.target_density / n,
        pred_density: total.pred_density / n,
        chance_f1_near: total.chance_f1_near / n,
        chance_chamfer: total.chance_chamfer / n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDE: usize = 32;

    fn blank() -> Vec<u8> {
        vec![0u8; SIDE * SIDE]
    }

    fn vertical_line(x: usize) -> Vec<u8> {
        let mut bits = blank();
        for y in 1..SIDE - 1 {
            bits[y * SIDE + x] = 1;
        }
        bits
    }

    #[test]
    fn distance_is_zero_on_ink_and_counts_pixels() {
        let d = distance_to_ink(&vertical_line(16), SIDE);
        assert_eq!(d[10 * SIDE + 16], 0.0);
        assert!((d[10 * SIDE + 17] - 1.0).abs() < 1e-6);
        assert!((d[10 * SIDE + 15] - 1.0).abs() < 1e-6);
        // Diagonal neighbour is sqrt(2) in this approximation.
        assert!((d[10 * SIDE + 17] - 1.0).abs() < 1e-6);
        assert!(d[10 * SIDE] > 15.0);
    }

    #[test]
    fn identical_drawings_score_one() {
        let line = vertical_line(16);
        let s = scores(&line, &line, SIDE);
        assert!((s.f1_near - 1.0).abs() < 1e-6);
        assert_eq!(s.chamfer, 0.0);
    }

    #[test]
    fn one_pixel_shift_is_forgiven_but_ber_would_not() {
        // This is the whole point: a perfect stroke off by one scores F1 0 by
        // the per-pixel rule and 1 here.
        let target = vertical_line(16);
        let pred = vertical_line(17);
        let s = scores(&target, &pred, SIDE);
        assert!((s.f1_near - 1.0).abs() < 1e-6, "f1 {}", s.f1_near);
        assert!((s.chamfer - 1.0).abs() < 0.1, "chamfer {}", s.chamfer);
    }

    #[test]
    fn empty_prediction_misses_everything() {
        let s = scores(&vertical_line(16), &blank(), SIDE);
        assert_eq!(s.f1_near, 0.0);
        assert_eq!(s.f1_far, 0.0);
        assert!(s.chamfer > 1.0);
        assert_eq!(s.pred_density, 0.0);
    }

    #[test]
    fn blank_image_distances_are_finite() {
        // The two-pass transform leaves the sentinel behind when there is no
        // ink at all; it must be clamped to the diagonal, not reported as 1e9.
        let cap = SIDE as f32 * std::f32::consts::SQRT_2;
        let d = distance_to_ink(&blank(), SIDE);
        assert!(d.iter().all(|v| (*v - cap).abs() < 1e-3), "unclamped {:?}", d[0]);
        let s = scores(&vertical_line(16), &blank(), SIDE);
        assert!((s.chamfer - cap).abs() < 1e-3, "chamfer {}", s.chamfer);
        assert!(s.chamfer < 100.0, "chamfer {} is not in pixels", s.chamfer);
    }

    #[test]
    fn filling_the_image_is_punished_by_precision() {
        // Perfect recall, terrible precision: 0.16 against 1.0 for a correctly
        // placed stroke, so the radius does not make the metric trivial.
        let pred = vec![1u8; SIDE * SIDE];
        let s = scores(&vertical_line(16), &pred, SIDE);
        assert!(
            s.f1_near < 0.2,
            "f1 {} (precision must reject a full canvas)",
            s.f1_near
        );
        assert_eq!(s.pred_density, 1.0);
    }

    #[test]
    fn chance_ink_scores_far_below_a_real_drawing() {
        let target = vertical_line(16);
        let perfect = scores(&target, &target, SIDE);
        let mut rng = 12345u64;
        let mut spray = vec![0u8; SIDE * SIDE];
        for slot in spray.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if rng % 100 < 15 {
                *slot = 1;
            }
        }
        let noise = scores(&target, &spray, SIDE);
        assert!(perfect.f1_near > 0.99);
        assert!(
            noise.f1_near < 0.5,
            "spray scored {} against a single line",
            noise.f1_near
        );
    }

    #[test]
    fn empty_target_and_prediction_are_perfect() {
        let s = scores(&blank(), &blank(), SIDE);
        assert_eq!(s.f1_near, 1.0);
        assert_eq!(s.chamfer, 0.0);
    }
}
