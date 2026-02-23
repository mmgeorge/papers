pub mod params;
pub mod server;

/// Start the MCP server on stdio. Blocks until the connection closes.
pub async fn start_stdio() -> Result<(), Box<dyn std::error::Error>> {
    use rmcp::ServiceExt;
    let server = server::PapersMcp::new().await;
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
