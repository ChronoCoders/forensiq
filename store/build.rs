fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // DuckDB 1.x on Windows uses the Restart Manager API to detect file locks.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows" {
        println!("cargo:rustc-link-lib=rstrtmgr");
    }
}
