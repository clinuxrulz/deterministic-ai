//! Benchmark the training hot loop (candidate gather + weighted NAND sum) on
//! the GPU, and the same computation on the CPU, so we can decide whether a
//! GPU trainer is worth building for this device.
//!
//! The GPU path is deliberately minimal: one compute pipeline, host-visible
//! storage buffers, and a pre-recorded command buffer submitted N times.

use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use ash::vk;

pub struct Options {
    pub gates: usize,
    pub candidates: usize,
    pub samples: usize,
    pub wires: usize,
    pub iterations: usize,
}

impl Default for Options {
    fn default() -> Self {
        let env = |name: &str, fallback: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(fallback)
        };
        Self {
            gates: env("BENCH_GATES", 4096),
            candidates: env("BENCH_CANDIDATES", 16),
            samples: env("BENCH_SAMPLES", 256),
            wires: env("BENCH_WIRES", 8192),
            iterations: env("BENCH_ITERS", 20),
        }
    }
}

struct Gpu {
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
}

impl Gpu {
    fn new(loader: &str) -> Result<Self> {
        let entry = unsafe { ash::Entry::load_from(loader) }
            .map_err(|e| anyhow!("loading {loader}: {e}"))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"xorae-gpu-probe")
            .api_version(vk::make_api_version(0, 1, 0, 0));
        let instance = unsafe {
            entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
        }
        .context("creating instance")?;
        let devices = unsafe { instance.enumerate_physical_devices() }?;
        // Prefer a real GPU over llvmpipe/lavapipe.
        let physical = devices
            .iter()
            .copied()
            .max_by_key(|d| {
                let p = unsafe { instance.get_physical_device_properties(*d) };
                match p.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                    _ => 0,
                }
            })
            .ok_or_else(|| anyhow!("no Vulkan device"))?;
        let props = unsafe { instance.get_physical_device_properties(physical) };
        println!(
            "gpu: {} ({:?})",
            props
                .device_name_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "?".into()),
            props.device_type
        );

        let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = queues
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow!("no compute queue"))? as u32;
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
                None,
            )
        }
        .context("creating device")?;
        let queue = unsafe { device.get_device_queue(family, 0) };
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )
        }?;
        Ok(Self {
            device,
            queue,
            command_pool,
            _entry: entry,
            instance,
            physical,
        })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A host-visible storage buffer plus its mapped pointer.
struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    len: usize,
    ptr: *mut u8,
}

impl Gpu {
    fn buffer(&self, len: usize) -> Result<Buf> {
        let size = (len * 4) as u64;
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let props = unsafe { self.instance.get_physical_device_memory_properties(self.physical) };
        let index = (0..props.memory_type_count)
            .find(|&i| {
                let flags = props.memory_types[i as usize].property_flags;
                req.memory_type_bits & (1 << i) != 0
                    && flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            })
            .ok_or_else(|| anyhow!("no host-visible memory"))?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(index),
                None,
            )
        }?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }?;
        let ptr = unsafe {
            self.device
                .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())?
        } as *mut u8;
        Ok(Buf {
            buffer,
            memory,
            len,
            ptr,
        })
    }
}

fn load_spirv(source: &str) -> Result<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| anyhow!("wgsl: {e:?}"))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .map_err(|e| anyhow!("wgsl validation: {e:?}"))?;
    let options = naga::back::spv::Options::default();
    naga::back::spv::write_vec(&module, &info, &options, None).map_err(|e| anyhow!("spirv: {e:?}"))
}

fn pipeline(
    gpu: &Gpu,
    spv: &[u32],
    entry: &str,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::Pipeline> {
    let module = unsafe {
        gpu.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(spv),
            None,
        )
    }?;
    let entry_name = std::ffi::CString::new(entry)?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry_name);
    let layout = unsafe {
        gpu.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),
            None,
        )
    }?;
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipes = unsafe {
        gpu.device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .map_err(|(_, e)| anyhow!("pipeline: {e}"))?;
    unsafe { gpu.device.destroy_shader_module(module, None) };
    Ok(pipes[0])
}

/// Run `iterations` dispatches of a pre-recorded command buffer.
fn time_dispatches(gpu: &Gpu, cmd: vk::CommandBuffer, iterations: usize) -> Result<f64> {
    let fence = unsafe { gpu.device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
    let cmds = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&cmds);
    // Warm-up.
    unsafe {
        gpu.device.queue_submit(gpu.queue, &[submit], fence)?;
        gpu.device.wait_for_fences(&[fence], true, u64::MAX)?;
        gpu.device.reset_fences(&[fence])?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        unsafe {
            gpu.device.queue_submit(gpu.queue, &[submit], fence)?;
            gpu.device.wait_for_fences(&[fence], true, u64::MAX)?;
            gpu.device.reset_fences(&[fence])?;
        }
    }
    let elapsed = started.elapsed().as_secs_f64() / iterations as f64;
    unsafe { gpu.device.destroy_fence(fence, None) };
    Ok(elapsed)
}

fn setup(
    gpu: &Gpu,
    spv: &[u32],
    entry: &str,
    buffers: &[vk::Buffer],
    groups: u32,
) -> Result<(vk::CommandBuffer, vk::Pipeline, vk::DescriptorSetLayout, vk::PipelineLayout)> {
    let bindings: Vec<_> = (0..buffers.len())
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let set_layout = unsafe {
        gpu.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }?;
    let pipe = pipeline(gpu, spv, entry, set_layout)?;
    let layout = unsafe {
        gpu.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),
            None,
        )
    }?;
    let sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(buffers.len() as u32)];
    let pool = unsafe {
        gpu.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&sizes),
            None,
        )
    }?;
    let sets = unsafe {
        gpu.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&[set_layout]),
        )
    }?;
    let infos: Vec<vk::DescriptorBufferInfo> = buffers
        .iter()
        .map(|b| {
            vk::DescriptorBufferInfo::default()
                .buffer(*b)
                .offset(0)
                .range(vk::WHOLE_SIZE)
        })
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(sets[0])
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
    unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };

    let cmd = unsafe {
        gpu.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(gpu.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }?[0];
    unsafe {
        gpu.device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
        )?;
        gpu.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        gpu.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &sets,
            &[],
        );
        gpu.device.cmd_dispatch(cmd, groups, 1, 1);
        gpu.device.end_command_buffer(cmd)?;
    }
    Ok((cmd, pipe, set_layout, layout))
}

pub fn run(loader: &str, opts: &Options) -> Result<()> {
    let gpu = Gpu::new(loader)?;
    let total = opts.gates * opts.samples;

    // --- gather kernel: the trainer's actual inner loop -------------------
    let mut cand = vec![0u32; opts.gates * opts.candidates];
    let mut p = vec![0f32; opts.gates * opts.candidates];
    let mut x = vec![0f32; opts.wires * opts.samples];
    let mut state = 12345u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as u32
    };
    for c in cand.iter_mut() {
        *c = next() % opts.wires as u32;
    }
    for v in p.iter_mut() {
        *v = 0.5;
    }
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i * 2654435761) % 1000) as f32 / 1000.0;
    }

    let src = format!(
        r#"
@group(0) @binding(0) var<storage, read> cand: array<u32>;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&y)) {{ return; }}
    let g = idx / {samples}u;
    let s = idx % {samples}u;
    var acc = 0.0;
    for (var c = 0u; c < {candidates}u; c = c + 1u) {{
        acc = acc + p[g * {candidates}u + c] * x[cand[g * {candidates}u + c] * {samples}u + s];
    }}
    y[idx] = 1.0 - acc * acc;
}}
"#,
        samples = opts.samples,
        candidates = opts.candidates,
    );
    let spv = load_spirv(&src)?;

    let b_cand = gpu.buffer(cand.len())?;
    let b_p = gpu.buffer(p.len())?;
    let b_x = gpu.buffer(x.len())?;
    let b_y = gpu.buffer(total)?;
    unsafe {
        std::ptr::copy_nonoverlapping(cand.as_ptr() as *const u8, b_cand.ptr, cand.len() * 4);
        std::ptr::copy_nonoverlapping(p.as_ptr() as *const u8, b_p.ptr, p.len() * 4);
        std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, b_x.ptr, x.len() * 4);
    }
    let bufs = [b_cand.buffer, b_p.buffer, b_x.buffer, b_y.buffer];
    let groups = (total as u32).div_ceil(64);
    let (cmd, _pipe, _sl, _pl) = setup(&gpu, &spv, "gather", &bufs, groups)?;
    let secs = time_dispatches(&gpu, cmd, opts.iterations)?;
    let flops = 2.0 * (opts.gates * opts.candidates * opts.samples) as f64;
    println!(
        "gpu gather: {:.1} ms/iter  {:.1} GFLOP/s  ({} gates x {} cands x {} samples)",
        secs * 1e3,
        flops / secs / 1e9,
        opts.gates,
        opts.candidates,
        opts.samples
    );

    // Read back one value so the compiler cannot elide the work.
    let first = unsafe { *(b_y.ptr as *const f32) };
    println!("gpu sample output: {first:.4}");

    // --- same gather on the CPU, matching the trainer's layout ------------
    use rayon::prelude::*;
    let mut y_cpu = vec![0f32; total];
    let started = Instant::now();
    let cpu_iters = 3;
    for _ in 0..cpu_iters {
        y_cpu
            .par_chunks_mut(opts.samples)
            .enumerate()
            .for_each(|(g, row)| {
                for (s, out) in row.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for c in 0..opts.candidates {
                        let src = cand[g * opts.candidates + c] as usize;
                        acc += p[g * opts.candidates + c] * x[src * opts.samples + s];
                    }
                    *out = 1.0 - acc * acc;
                }
            });
    }
    let cpu_secs = started.elapsed().as_secs_f64() / cpu_iters as f64;
    println!(
        "cpu gather: {:.1} ms/iter  {:.1} GFLOP/s  ({} threads)",
        cpu_secs * 1e3,
        flops / cpu_secs / 1e9,
        rayon::current_num_threads()
    );

    // Sanity: the two must agree.
    let max_diff = y_cpu
        .iter()
        .zip(unsafe { std::slice::from_raw_parts(b_y.ptr as *const f32, total) })
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max |gpu - cpu| = {max_diff:.6}");
    println!("speedup: {:.2}x", cpu_secs / secs);
    Ok(())
}
