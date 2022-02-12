pub mod seq0;
pub use seq0::apply_diffusion as seq0;

pub mod seq1;
pub use seq1::apply_diffusion as seq1;

pub mod seq2;
pub use seq2::apply_diffusion as seq2;

pub mod seq;
pub use seq::apply_diffusion as seq;

pub mod par;
pub use par::apply_diffusion as par;
