use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let workspace_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let cache_dir = workspace_root.join(".cache");
    fs::create_dir_all(&cache_dir).unwrap();

    download_pdfium_if_needed(&cache_dir);

    #[cfg(target_os = "windows")]
    download_ort_if_needed(&cache_dir);

    // Touch empty sentinel files for cross-platform dist compatibility.
    // cargo-dist include is static (no per-platform conditionals), so we ensure
    // all listed files exist — empty files won't be loaded on wrong platforms.
    for name in &[
        "pdfium.dll",
        "libpdfium.dylib",
        "onnxruntime.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_shared.dll",
    ] {
        let p = cache_dir.join(name);
        if !p.exists() {
            fs::File::create(&p).ok();
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn download_pdfium_if_needed(cache_dir: &PathBuf) {
    let lib_filename = pdfium_lib_filename();
    let dest = cache_dir.join(lib_filename);
    if dest.exists() {
        return;
    }

    let url = pdfium_download_url();
    let archive_path = pdfium_archive_path();

    eprintln!("build.rs: Downloading pdfium from {url}...");

    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .expect("Failed to create HTTP client");

    let response = client.get(&url).send().expect("Pdfium download failed");
    if !response.status().is_success() {
        panic!("Pdfium download failed: HTTP {}", response.status());
    }

    let bytes = response.bytes().expect("Failed to read pdfium archive");

    let decoder = flate2::read::GzDecoder::new(bytes.as_ref());
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries().expect("Failed to read tar entries") {
        let mut entry = entry.expect("Failed to read tar entry");
        let path = entry.path().expect("Invalid path in archive");

        if path.to_str().map_or(false, |p| p == archive_path) {
            let tmp = dest.with_extension("tmp");
            let mut file = fs::File::create(&tmp).expect("Failed to create temp file");
            std::io::copy(&mut entry, &mut file).expect("Failed to extract pdfium");
            file.flush().unwrap();
            drop(file);
            fs::rename(&tmp, &dest).expect("Failed to rename pdfium");
            eprintln!("build.rs: Pdfium extracted to {}", dest.display());
            return;
        }
    }

    panic!("Could not find {archive_path} in pdfium archive");
}

fn pdfium_download_url() -> String {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.tgz"
            .to_string()
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-arm64.tgz"
            .to_string()
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-mac-x64.tgz"
            .to_string()
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-mac-arm64.tgz"
            .to_string()
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x64.tgz"
            .to_string()
    }
}

fn pdfium_archive_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "bin/pdfium.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "lib/libpdfium.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "lib/libpdfium.so"
    }
}

fn pdfium_lib_filename() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "pdfium.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libpdfium.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "libpdfium.so"
    }
}

#[cfg(target_os = "windows")]
fn download_ort_if_needed(cache_dir: &PathBuf) {
    let marker = cache_dir.join("onnxruntime.dll");
    if marker.exists() {
        return;
    }

    let url = "https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-win-x64-gpu-1.24.2.zip";
    eprintln!("build.rs: Downloading ORT from {url}...");

    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .expect("Failed to create HTTP client");

    let response = client.get(url).send().expect("ORT download failed");
    if !response.status().is_success() {
        panic!("ORT download failed: HTTP {}", response.status());
    }

    let bytes = response.bytes().expect("Failed to read ORT archive");

    let reader = std::io::Cursor::new(bytes.as_ref());
    let mut archive = zip::ZipArchive::new(reader).expect("Failed to open ORT zip");

    let dlls = [
        "onnxruntime.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_shared.dll",
    ];

    for dll_name in &dlls {
        // Find the entry — it's inside a subdirectory like onnxruntime-win-x64-gpu-1.24.2/lib/
        let entry_name = archive
            .file_names()
            .find(|n| n.ends_with(&format!("/lib/{dll_name}")))
            .map(|s| s.to_string());

        let entry_name = entry_name.unwrap_or_else(|| {
            panic!("Could not find {dll_name} in ORT archive");
        });

        let mut entry = archive.by_name(&entry_name).unwrap();
        let dest = cache_dir.join(dll_name);
        let tmp = dest.with_extension("tmp");
        let mut file = fs::File::create(&tmp).expect("Failed to create temp file");
        std::io::copy(&mut entry, &mut file).expect("Failed to extract ORT DLL");
        file.flush().unwrap();
        drop(file);
        fs::rename(&tmp, &dest).expect("Failed to rename ORT DLL");
        eprintln!("build.rs: Extracted {dll_name} to {}", dest.display());
    }
}
