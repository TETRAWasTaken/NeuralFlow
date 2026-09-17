#[cfg(feature = "xla")]
use xla::{XlaBuilder, XlaOp};

pub trait XlaTraceable {
    #[cfg(feature = "xla")]
    fn trace(&self, builder: &XlaBuilder, input: &XlaOp) -> Result<XlaOp, xla::Error>;
}
