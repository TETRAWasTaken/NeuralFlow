use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Gpu(usize),
    Tpu(usize),
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }

    pub fn is_tpu(&self) -> bool {
        matches!(self, Device::Tpu(_))
    }

    pub fn device_id(&self) -> Option<usize> {
        match self {
            Device::Cpu => None,
            Device::Gpu(id) | Device::Tpu(id) => Some(*id),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Gpu(id) => write!(f, "gpu:{}", id),
            Device::Tpu(id) => write!(f, "tpu:{}", id),
        }
    }
}
