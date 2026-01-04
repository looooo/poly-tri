//! PolyTri - Delaunay triangulation with constrained boundaries and hole removal.
//!
//! This crate provides a Delaunay triangulation implementation that supports:
//! - Constrained boundaries
//! - Hole removal
//! - Non-convex geometries

pub mod polytri;

#[cfg(feature = "python")]
pub mod python_bindings;

pub use polytri::{Point, PolyTri, PolyTriError};

// Python module is registered in python_bindings.rs
