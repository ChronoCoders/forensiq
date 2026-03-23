use ingestion::{ingest_file, verify_integrity};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("sample.bin");
    let evidence = ingest_file(path, "cli", "operator-1").await?;
    println!("Ingested: {} sha256={}", evidence.uuid, evidence.sha256);

    let ok = verify_integrity(&evidence)?;
    println!("Integrity OK: {ok}");

    Ok(())
}
