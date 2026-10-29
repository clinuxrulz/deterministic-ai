#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(|| leptos::view! { <xorae_web::app::App /> });
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("xorae-web is a WebAssembly app. Run `trunk serve` from crates/web.");
}
