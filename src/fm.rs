use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use pyo3::prelude::*;
use xxhash_rust::xxh3::xxh3_64;

#[pyclass]
pub struct FactorizationMachine {
    weights: Array1<f32>,
    factors: Array2<f32>,
    k: usize,
    n_bins: usize,
}

#[pymethods]
impl FactorizationMachine {
    #[new]
    pub fn new(k: usize, n_bins: usize) -> Self {
        FactorizationMachine {
            weights: Array1::<f32>::zeros(n_bins + 1), // +1 for the bias term
            factors: Array2::<f32>::random((n_bins, k), Uniform::new(-0.01, 0.01)),
            k,
            n_bins,
        }
    }

    fn hash_feature(&self, feature: &str) -> usize {
        (xxh3_64(feature.as_bytes()) as usize) % self.n_bins
    }

    pub fn predict(&self, features: Vec<(String, f32)>) -> f32 {
        let mut linear_term = self.weights[0];
        let mut interaction_term = 0.0;
        let mut feature_indices = vec![];

        for (feature, value) in features {
            let index = self.hash_feature(&feature);
            linear_term += self.weights[index + 1] * value;
            feature_indices.push((index, value));
        }

        for f in 0..self.k {
            let sum_square: f32 = feature_indices.iter().map(|&(index, value)| value * self.factors[[index, f]]).sum();
            let square_sum: f32 = feature_indices.iter().map(|&(index, value)| value.powi(2) * self.factors[[index, f]].powi(2)).sum();

            interaction_term += 0.5 * (sum_square.powi(2) - square_sum);
        }

        linear_term + interaction_term
    }

    pub fn train(&mut self, data: Vec<Vec<(String, f32)>>, target: Vec<f32>, learning_rate: f32, epochs: usize) {
        for _ in 0..epochs {
            for (features, &y) in data.iter().zip(target.iter()) {
                let y_pred = self.predict(features.clone());
                let error = y - y_pred;

                self.weights[0] += learning_rate * error;

                for (feature, value) in features {
                    let index = self.hash_feature(&feature);
                    self.weights[index + 1] += learning_rate * error * value;

                    for f in 0..self.k {
                        let v_if = self.factors[[index, f]];
                        let sum_x_v: f32 = features.iter()
                            .map(|(feat, val)| {
                                let idx = self.hash_feature(feat);
                                val * self.factors[[idx, f]]
                            })
                            .sum();
                        self.factors[[index, f]] += learning_rate * error * (value * (sum_x_v - value * v_if));
                    }
                }
            }
        }
    }
}

#[pyfunction]
pub fn create_factorization_machine(k: usize, n_bins: usize) -> FactorizationMachine {
    FactorizationMachine::new(k, n_bins)
}