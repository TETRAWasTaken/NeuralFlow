#[cfg(feature = "xla")]
use xla::{PjRtBuffer, PjRtClient, PjRtLoadedExecutable, XlaComputation};

#[cfg(feature = "xla")]
pub struct XlaCompiledStep {
    pub executable: PjRtLoadedExecutable,
}

#[cfg(feature = "xla")]
impl XlaCompiledStep {
    pub fn compile(client: &PjRtClient, computation: &XlaComputation) -> Result<Self, xla::Error> {
        let executable = client.compile(computation)?;
        Ok(Self { executable })
    }

    /// Execute the compiled step on device buffers
    pub fn execute(&self, inputs: &[&PjRtBuffer]) -> Result<Vec<PjRtBuffer>, xla::Error> {
        let mut results = self.executable.execute(inputs)?;
        if results.is_empty() {
            Ok(vec![])
        } else {
            Ok(results.remove(0))
        }
    }
}
