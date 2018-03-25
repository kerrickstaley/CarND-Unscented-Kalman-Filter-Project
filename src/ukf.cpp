#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // Let's assume a cyclist can get to 40 kmph in 5 seconds. That's 11.11 m/s, which means an acceleration of 2.22
  // m/s^2. Using the rule that sigma_accel = 0.5 * max_accel, we get std_a_ = 1.11
  std_a_ = 1.11;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // Let's assume a cyclist can execute a 90 degree turn in as little as one second, and that the angular velocity for
  // their turn looks like this: /\. Their average angular speed during this turn is 0.5*pi rad/s, so their peak speed
  // is pi rad/s, which they attain after 0.5 seconds. Hence their angular acceleration in this case is 2*pi rad/s^2.
  // Using the rule that sigma_accel = 0.5 * max_accel, we get std_yawdd_ = pi rad/s^2
  std_yawdd_ = 3.14;

  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;

    // set initial state
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
      double rho = meas_package.raw_measurements_(0),
             gamma = meas_package.raw_measurements_(1),
             rho_dot = meas_package.raw_measurements(2);

      x_ << rho * cos(gamma),  // pos_x
            rho * sin(gamma),  // pos_y
            0,                 // v_abs (curious what happens when we use fabs(rho_dot))
            0,                 // yaw_angle (curious what happens when we use gamma + (rho_dot < 0 ? pi : 0))
            0;                 // yaw_rate
    } else {
      double px = meas_package.raw_measurements_(0),
             py = meas_package.raw_measurements_(1);

      x_ << px, py, 0, 0, 0;
    }

    P_ << MatrixXd::Identity(5, 5);

    is_initialized_ = true;

    return;
  }

  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
