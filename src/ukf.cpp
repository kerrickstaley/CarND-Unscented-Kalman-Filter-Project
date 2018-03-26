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

  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  weights_ =  VectorXD(2 * n_aug_ + 1);
  weights_.fill(1 / (2.0 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // initialize H_laser_ and R_laser_
  H_laser_ = MatrixXD(2, n_x_);
  H_laser_.fillZero();
  H_laser_(0, 0) = 1;
  H_laser_(1, 1) = 1;

  R_laser_ = MatrixXD(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
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
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
      double px = meas_package.raw_measurements_(0),
             py = meas_package.raw_measurements_(1);

      x_ << px, py, 0, 0, 0;
    } else {
      double rho = meas_package.raw_measurements_(0),
             gamma = meas_package.raw_measurements_(1),
             rho_dot = meas_package.raw_measurements(2);

      x_ << rho * cos(gamma),  // pos_x
            rho * sin(gamma),  // pos_y
            0,                 // v_abs (curious what happens when we use fabs(rho_dot))
            0,                 // yaw_angle (curious what happens when we use gamma + (rho_dot < 0 ? pi : 0))
            0;                 // yaw_rate
    }

    P_ << MatrixXd::Identity(5, 5);

    is_initialized_ = true;

    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1.0e6;
  time_us = meas_package.timestamp_;

  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
    UpdateLidar(meas_package);
  } else {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // calculate augmented state mean vector
  VectorXD x_aug(n_aug_);
  x_aug.fillZero();
  x_aug.head(n_x_) = x_;

  // calculate augmented state covariance matrix
  MatrixXD P_aug(n_aug_, n_aug_);
  P_aug.fillZero();
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // calculate sigma points
  MatrixXD Xsig(n_aug_, 2 * n_aug_ + 1);
  MatrixXD P_aug_sqrt = P_aug.llt().matrixL();
  MatrixXD spread = sqrt(lambda + n_aug_) * P_aug_sqrt;

  Xsig.col(0) = x_aug;

  for (int i = 0; i < n_aug_; i++) {
    Xsig.col(1 + i) = x_aug + spread.col(i);
    Xsig.col(1 + n_aug_ + i) = x_aug - spread.col(i);
  }

  // run sigma points through prediction function
  for (int i = 0; i < 2 * n_aug + 1; i++) {
    double px = Xsig(0, i);
    double py = Xsig(1, i);
    double v = Xsig(2, i);
    double yaw = Xsig(3, i);
    double yaw_dot = Xsig(4, i);
    double nu_a = Xsig(5, i);
    double nu_yaw = Xsig(6, i);

    // calculate updated px and py
    // check for zero yaw_dot case
    if (fabs(yaw_dot) < 1e-6) {
      Xsig_pred_(0, i) = px + v * cos(yaw) * delta_t;
      Xsig_pred_(1, i) = py + v * sin(yaw) * delta_t;
    } else {
      Xsig_pred_(0, i) = v / yaw_dot
                         * (sin(yaw + yaw_dot * delta_t) - sin(yaw));
      Xsig_pred_(1, i) = v / yaw_dot
                         * (-cos(yaw + yaw_dot * delta_t) + cos(yaw));
    }
    Xsig_pred(0, i) += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
    Xsig_pred(1, i) += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;

    // calculate updated v
    Xsig_pred_(2, i) = v + delta_t * nu_a;

    // calculate updated yaw
    Xsig_pred_(3, i) = yaw + yaw_dot * delta_t + 0.5 * delta_t * delta_t * nu_yaw;

    // calculate updated yaw_dot
    Xsig_pred_(4, i) = yaw_dot + delta_t * nu_yaw;
  }

  // calculate updated state vector
  x_.fillZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // calculate updated state covariance matrix
  P_.fillZero();
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    VectorXD diff = Xsig_pred_.col(i) - x_;
    P_ += weights_(i) * diff * diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // lidar measurement is linear, so we just use the standard KF equations instead of UKF
  MatrixXD& H = H_laser_;
  MatrixXD& R = R_laser_;

  VectorXD y = meas_package.raw_measurements_ - H * x_;
  MatrixXD S = H * P_ * H.transpose() + R;

  MatrixXD K = P_ * H.transpose() * S.inverse();

  x_ += K * y;
  P_ -= K * H * P_;

  /**
  TODO:

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
