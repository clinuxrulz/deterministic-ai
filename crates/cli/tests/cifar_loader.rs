use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use xorae::dataset::{load_cifar10, Dataset, CIFAR_RECORD};
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
