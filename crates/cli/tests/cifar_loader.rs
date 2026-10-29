use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use xorae::dataset::{load_cifar10, load_cifar10_rgb2, Dataset, CIFAR_RECORD};
use xorae_core::preprocess::{BinarizeMode, PreprocessOptions};

fn temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("xorae-cifar-{label}-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_batch(path: &Path, records: &[(u8, u8)]) {
    let mut bytes = Vec::with_capacity(records.len() * CIFAR_RECORD);
    for &(label, value) in records {
        bytes.push(label);
        bytes.extend(std::iter::repeat_n(value, 32 * 32 * 3));
    }
    std::fs::write(path, bytes).unwrap();
}

/// The loader expects all five training batches to exist.
fn write_empty_batches(dir: &Path) {
    for i in 2..=5 {
        std::fs::write(dir.join(format!("data_batch_{i}.bin")), []).unwrap();
    }
}

#[test]
fn cifar_loader_filters_class_and_preserves_split() {
    let dir = temp_dir("loader");
    // Two cats and one non-cat in the train batch; one cat and one non-cat
    // in the test batch.
    write_batch(&dir.join("data_batch_1.bin"), &[(3, 255), (0, 0), (3, 0)]);
    write_batch(&dir.join("test_batch.bin"), &[(3, 255), (5, 0)]);
    write_empty_batches(&dir);

    let opts = PreprocessOptions {
        size: 8,
        mode: BinarizeMode::Threshold,
        threshold: 128,
    };
    let dataset = load_cifar10(&dir, 3, opts).unwrap();
    assert_eq!(dataset.n, 3);
    assert_eq!(dataset.cols(), 64);
    assert_eq!(dataset.is_test, vec![false, false, true]);

    // First cat is all white, second is all black.
    assert!(dataset.row_bits(0).iter().all(|&b| b == 1));
    assert!(dataset.row_bits(1).iter().all(|&b| b == 0));
    assert!(dataset.row_gray(0).iter().all(|&v| v == 255));
    assert!(dataset.row_gray(1).iter().all(|&v| v == 0));

    let (train, val, test) = dataset.split_indices(7);
    assert_eq!(test, vec![2]);
    assert_eq!(train.len() + val.len(), 2);
    assert!(!train.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cifar_loader_uses_planar_channel_order() {
    let dir = temp_dir("planar");
    let mut record = Vec::with_capacity(CIFAR_RECORD);
    record.push(3); // cat
    record.extend(std::iter::repeat_n(255u8, 32 * 32)); // red plane
    record.extend(std::iter::repeat_n(0u8, 32 * 32)); // green plane
    record.extend(std::iter::repeat_n(0u8, 32 * 32)); // blue plane
    std::fs::write(dir.join("data_batch_1.bin"), record).unwrap();
    write_empty_batches(&dir);
    std::fs::write(dir.join("test_batch.bin"), []).unwrap();

    let opts = PreprocessOptions {
        size: 8,
        mode: BinarizeMode::Threshold,
        threshold: 128,
    };
    let dataset = load_cifar10(&dir, 3, opts).unwrap();
    assert_eq!(dataset.n, 1);
    // Luma of pure red is 0.299 * 255 = 76. Interpreting the planes as
    // interleaved RGB would produce mostly 255 here.
    assert!(
        dataset.row_gray(0).iter().all(|&value| value == 76),
        "expected planar red luma 76, got {:?}",
        &dataset.row_gray(0)[..8]
    );
    assert!(dataset.row_bits(0).iter().all(|&bit| bit == 0));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dataset_roundtrips_through_disk() {
    let dir = temp_dir("roundtrip");
    write_batch(&dir.join("data_batch_1.bin"), &[(3, 255), (3, 0)]);
    write_batch(&dir.join("test_batch.bin"), &[(3, 255)]);
    write_empty_batches(&dir);

    let opts = PreprocessOptions {
        size: 8,
        mode: BinarizeMode::Dither,
        threshold: 128,
    };
    let dataset: Dataset = load_cifar10(&dir, 3, opts).unwrap();
    let path = dir.join("cats.bin");
    dataset.save(&path).unwrap();
    let loaded = Dataset::load(&path).unwrap();

    assert_eq!(loaded.size, dataset.size);
    assert_eq!(loaded.n, dataset.n);
    assert_eq!(loaded.is_test, dataset.is_test);
    assert_eq!(loaded.gray, dataset.gray);
    assert_eq!(loaded.bits, dataset.bits);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Build a one-image CIFAR batch whose channels are constant.
fn write_constant_cat(dir: &Path, rgb: [u8; 3]) {
    let mut record = vec![3u8];
    for channel in rgb {
        record.extend(std::iter::repeat_n(channel, 32 * 32));
    }
    std::fs::write(dir.join("data_batch_1.bin"), record).unwrap();
    std::fs::write(dir.join("test_batch.bin"), []).unwrap();
    write_empty_batches(dir);
}

#[test]
fn rgb2_loader_quantises_channels_into_six_planes() {
    let dir = temp_dir("rgb2");
    // R = 255 -> level 3 (hi and lo), G = 128 -> level 2 (hi only),
    // B = 0 -> level 0 (neither).
    write_constant_cat(&dir, [255, 128, 0]);

    let dataset = load_cifar10_rgb2(&dir, 3, 2).unwrap();
    assert_eq!(dataset.channels, 6);
    assert_eq!(dataset.bits_per_pixel, 1);
    assert_eq!(dataset.cols(), 4 * 6);

    let bits = dataset.row_bits(0);
    assert_eq!(bits.len(), 24);
    assert!(bits[0..4].iter().all(|&b| b == 1), "R_hi");
    assert!(bits[4..8].iter().all(|&b| b == 1), "R_lo");
    assert!(bits[8..12].iter().all(|&b| b == 1), "G_hi");
    assert!(bits[12..16].iter().all(|&b| b == 0), "G_lo");
    assert!(bits[16..24].iter().all(|&b| b == 0), "B_hi/B_lo");

    // The dataset cache must preserve the mode, channel count and bits.
    let path = dir.join("rgb2.bin");
    dataset.save(&path).unwrap();
    let loaded = Dataset::load(&path).unwrap();
    assert_eq!(loaded.color_mode, dataset.color_mode);
    assert_eq!(loaded.channels, 6);
    assert_eq!(loaded.bits_per_pixel, 1);
    assert_eq!(loaded.row_bits(0), bits);

    let _ = std::fs::remove_dir_all(&dir);
}
