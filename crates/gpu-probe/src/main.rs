//! Minimal Vulkan probe: can this machine, and this Termux install, open a
//! Vulkan loader and enumerate a compute-capable device?
//!
//! Deliberately uses `ash`'s dynamic loading (libloading) instead of linking,
//! because Android ships `libvulkan.so` without the usual `.so.1` soname.

mod bench;

use anyhow::{anyhow, Context, Result};
use ash::vk;

fn load_entry() -> Result<ash::Entry> {
    // Allow forcing a specific loader, e.g. the Android system one:
    //   XORAE_VULKAN_LIB=/system/lib64/libvulkan.so gpu-probe
    if let Ok(path) = std::env::var("XORAE_VULKAN_LIB") {
        println!("loader: {path} (forced)");
        return unsafe { ash::Entry::load_from(&path) }
            .map_err(|e| anyhow!("loading {path}: {e}"));
    }
    // Android/Termux exposes the loader as `libvulkan.so`; desktop as `.so.1`.
    let candidates = [
        "libvulkan.so",
        "libvulkan.so.1",
        "/system/lib64/libvulkan.so",
        "/vendor/lib64/libvulkan.so",
    ];
    let mut errors = Vec::new();
    for path in candidates {
        match unsafe { ash::Entry::load_from(path) } {
            Ok(entry) => {
                println!("loader: {path}");
                return Ok(entry);
            }
            Err(e) => errors.push(format!("{path}: {e}")),
        }
    }
    Err(anyhow!("no Vulkan loader found:\n  {}", errors.join("\n  ")))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--bench") {
        let loader = std::env::var("XORAE_VULKAN_LIB")
            .unwrap_or_else(|_| "/system/lib64/libvulkan.so".to_string());
        return bench::run(&loader, &bench::Options::default());
    }

    let entry = load_entry()?;
    let version = unsafe { entry.try_enumerate_instance_version() };
    println!("instance version: {version:?}");

    let app = vk::ApplicationInfo::default()
        .application_name(c"xorae-gpu-probe")
        .api_version(vk::make_api_version(0, 1, 0, 0));
    let create = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&create, None) }
        .context("creating a Vulkan instance")?;

    let devices = unsafe { instance.enumerate_physical_devices() }
        .context("enumerating physical devices")?;
    if devices.is_empty() {
        println!("no Vulkan devices");
        return Ok(());
    }

    for (i, device) in devices.iter().enumerate() {
        let props = unsafe { instance.get_physical_device_properties(*device) };
        let name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "<unnamed>".into());
        let queues = unsafe { instance.get_physical_device_queue_family_properties(*device) };
        let compute = queues
            .iter()
            .filter(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .count();
        let memory: u64 = props
            .limits
            .max_compute_work_group_count
            .iter()
            .map(|&v| v as u64)
            .product();
        println!(
            "device {i}: {name}\n  type {:?}, api {}.{}.{}, vendor 0x{:04x}\n  compute queues: {compute}/{}\n  max workgroups: {memory}\n  max workgroup size: [{}, {}, {}]",
            props.device_type,
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version),
            props.vendor_id,
            queues.len(),
            props.limits.max_compute_work_group_size[0],
            props.limits.max_compute_work_group_size[1],
            props.limits.max_compute_work_group_size[2],
        );
    }
    unsafe { instance.destroy_instance(None) };
    Ok(())
}
