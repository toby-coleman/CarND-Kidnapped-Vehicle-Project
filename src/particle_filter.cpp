/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cfloat>
#include <math.h>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set number of particles
	num_particles = 30;
	// Initialise particles
	std::default_random_engine gen;
	std::normal_distribution<double> x_distribution(x, std[0]);
	std::normal_distribution<double> y_distribution(y, std[1]);
	std::normal_distribution<double> theta_distribution(theta, std[2]);

	for (int p = 0; p < num_particles; p++) {
		// Create a particle
		Particle particle;
		particle.id = p;
		particle.x = x_distribution(gen);
		particle.y = y_distribution(gen);
		particle.theta = theta_distribution(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(particle.weight);
	}

	// Particle filter is now initialised
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;
	std::normal_distribution<double> x_noise(0, std_pos[0]);
	std::normal_distribution<double> y_noise(0, std_pos[1]);
	std::normal_distribution<double> theta_noise(0, std_pos[2]);

	// Update each particle
	for (int p = 0; p < num_particles; p++) {
		const double x_0 = particles[p].x;
		const double y_0 = particles[p].y;
		const double theta_0 = particles[p].theta;
		if (fabs(yaw_rate) < 1e-5) {
			// Yaw rate is zero
			const double v_dt = velocity * delta_t;
			particles[p].x = x_0 + v_dt * cos(theta_0) + x_noise(gen);
			particles[p].y = y_0 + v_dt * sin(theta_0) + y_noise(gen);
			particles[p].theta = theta_0 + theta_noise(gen);
		} else {
			// Yaw rate is non-zero
			const double theta_plus = theta_0 + yaw_rate * delta_t;
			const double v_by_yaw_rate = velocity / yaw_rate;
			particles[p].x = x_0 + (sin(theta_plus) - sin(theta_0)) * v_by_yaw_rate + x_noise(gen);
			particles[p].y = y_0 + (cos(theta_0) - cos(theta_plus)) * v_by_yaw_rate + y_noise(gen);
			particles[p].theta = theta_plus + theta_noise(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// Loop over each observation
	for (int o = 0; o < observations.size(); o++) {
		double smallest_distance = DBL_MAX;
		// Loop over each predicted measurement
		for (int p = 0; p < predicted.size(); p++) {
			// Calculate squared distance
			const double dist = pow(predicted[p].x - observations[o].x, 2) +
													pow(predicted[p].y - observations[o].y, 2);
			if (dist < smallest_distance) {
				smallest_distance = dist;
				observations[o].id = p;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	const double squared_range = sensor_range * sensor_range;
	const double squared_sigma = std_landmark[0] * std_landmark[0];
	const double norm = 1.0 / (2.0 * M_PI * squared_sigma);

	// Update each particle
	for (int p = 0; p < num_particles; p++) {
		const double x = particles[p].x;
		const double y = particles[p].y;
		const double theta = particles[p].theta;
		// Build vector of observations in map coordinates
		std::vector<LandmarkObs> obs_t;
		for (int o = 0; o < observations.size(); o++) {
			LandmarkObs transformed;
			transformed.x = observations[o].x * cos(theta) -
											observations[o].y * sin(theta) +
											x;
			transformed.y = observations[o].x * sin(theta) +
											observations[o].y * cos(theta) +
											y;
			transformed.id = observations[o].id;
			obs_t.push_back(transformed);
		}
		// Build vector of predicted observations (those in sensor_range)
		std::vector<LandmarkObs> predicted;
		for (int m = 0; m < map_landmarks.landmark_list.size(); m++) {
			const double squared_dist = pow(map_landmarks.landmark_list[m].x_f - x, 2) +
																	pow(map_landmarks.landmark_list[m].y_f - y, 2);
			if (squared_dist <= squared_range) {
				// In range
				LandmarkObs l;
				l.x = map_landmarks.landmark_list[m].x_f;
				l.y = map_landmarks.landmark_list[m].y_f;
				l.id = map_landmarks.landmark_list[m].id_i;
				predicted.push_back(l);
			}
		}
		// Don't update the weight if there are no predicted observations
		if (predicted.size() == 0) {
			weights[p] = particles[p].weight;
			continue;
		}
		// Assign each observation to nearest landmark on map
		dataAssociation(predicted, obs_t);
		// Compute weight
		double w = 1.0;
		for (int i = 0; i < obs_t.size(); i++) {
			// Compute x - mu_x, y - mu_y
			const double x_diff = obs_t[i].x - predicted[obs_t[i].id].x;
			const double y_diff = obs_t[i].y - predicted[obs_t[i].id].y;
			// Update w (simplified formula because correlation = 0)
			w *= norm * exp(-0.5 * (pow(x_diff, 2) / squared_sigma +
															pow(y_diff, 2) / squared_sigma));
		}
		// Update particle weight
		particles[p].weight = w;
		weights[p] = w;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> new_particles;
	std::default_random_engine gen;
	std::discrete_distribution<> dist(weights.begin(), weights.end());

	// Resample particles
	for (int p = 0; p < num_particles; p++) {
		new_particles.push_back(particles[dist(gen)]);
	}
	particles = new_particles;
	// Copy over weights
	for (int p = 0; p < num_particles; p++) {
		weights[p] = particles[p].weight;
	}
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
