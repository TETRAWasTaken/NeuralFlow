#[cfg(feature = "xla")]
use crate::tensor::Device;

#[cfg(feature = "xla")]
pub struct XlaClient {
    pub client: xla::PjRtClient,
    pub device: Device,
}

#[cfg(feature = "xla")]
impl XlaClient {
    pub fn cpu() -> Result<Self, xla::Error> {
        let client = xla::PjRtClient::cpu()?;
        Ok(Self {
            client,
            device: Device::Cpu,
        })
    }

    pub fn gpu(gpu_id: usize, memory_fraction: f64) -> Result<Self, xla::Error> {
        let client = xla::PjRtClient::gpu(memory_fraction, false)?;
        Ok(Self {
            client,
            device: Device::Gpu(gpu_id),
        })
    }

    pub fn tpu(tpu_id: usize) -> Result<Self, xla::Error> {
        let client = xla::PjRtClient::tpu()?;
        Ok(Self {
            client,
            device: Device::Tpu(tpu_id),
        })
    }

    pub fn for_device(device: Device) -> Result<Self, xla::Error> {
        match device {
            Device::Cpu => Self::cpu(),
            Device::Gpu(id) => Self::gpu(id, 0.9),
            Device::Tpu(id) => Self::tpu(id),
        }
    }

    pub fn device_name(&self) -> String {
        format!("{}", self.device)
    }

    pub fn addressable_devices(&self) -> Vec<String> {
        self.client
            .addressable_devices()
            .iter()
            .map(|d| format!("{:?}", d))
            .collect()
    }
}
