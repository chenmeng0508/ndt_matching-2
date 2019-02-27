/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 Localization program using Normal Distributions Transform

 Yuki KITSUKAWA
 */

#ifndef NDT_MATCHING_H_
#define NDT_MATCHING_H_

#include <pthread.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>


#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>


#include <ndt_cpu/NormalDistributionsTransform.h>
#include <pcl/registration/ndt.h>
#ifdef CUDA_FOUND
#include <ndt_gpu/NormalDistributionsTransform.h>
#endif
#ifdef USE_PCL_OPENMP
#include <pcl_omp_registration/ndt_.h>
#endif

#define PREDICT_POSE_THRESHOLD 0.5

#define Wa 0.4
#define Wb 0.3
#define Wc 0.3

class NDTMatching
{

public:
  NDTMatching();

private:
  void map_callback(const sensor_msgs::PointCloud2::ConstPtr &input);
  void points_callback(const sensor_msgs::PointCloud2::ConstPtr &input);
  void imu_calc(ros::Time current_time);
  void imu_odom_calc(ros::Time current_time);
  void odom_calc(ros::Time current_time);

private:
  struct pose
  {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
  };

  enum class MethodType
  {
    PCL_GENERIC = 0,
    PCL_ANH = 1,
    PCL_ANH_GPU = 2,
    PCL_OPENMP = 3,
  };

  MethodType method_type_ = MethodType::PCL_ANH;
  pose initial_pose_, predict_pose, predict_pose_imu_, predict_pose_odom_, predict_pose_imu_odom_, previous_pose_,
      ndt_pose_, current_pose_, current_pose_imu_, current_pose_odom_, current_pose_imu_odom_;
  double offset_imu_x_, offset_imu_y_, offset_imu_z_, offset_imu_roll_, offset_imu_pitch_, offset_imu_yaw_;
  double offset_odom_x_, offset_odom_y_, offset_odom_z_, offset_odom_roll_, offset_odom_pitch_, offset_odom_yaw_;
  double offset_imu_odom_x_, offset_imu_odom_y_, offset_imu_odom_z_, offset_imu_odom_roll_, offset_imu_odom_pitch_, offset_imu_odom_yaw_;
  // Can't load if typed "pcl::PointCloud<pcl::PointXYZRGB> map, add;"
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_ptr_;
  // If the map is loaded, map_loaded will be 1.
  int map_loaded_ = 0;
  int use_gnss_ = 1;
  int init_pos_set_ = 0;
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_;
  cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> anh_ndt_;
#ifdef CUDA_FOUND
  std::shared_ptr<gpu::GNormalDistributionsTransform> anh_gpu_ndt_ptr_ =
      std::make_shared<gpu::GNormalDistributionsTransform>();
#endif
#ifdef USE_PCL_OPENMP
  pcl_omp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> omp_ndt_;
#endif
  // Default values
  int max_iter_ = 30;       // Maximum iterations
  float ndt_res_ = 1.0;     // Resolution
  double step_size_ = 0.1;  // Step size
  double trans_eps_ = 0.01; // Transformation epsilon
  ros::Publisher predict_pose_pub_;
  geometry_msgs::PoseStamped predict_pose_msg_;
  ros::Publisher predict_pose_imu_pub_;
  geometry_msgs::PoseStamped predict_pose_imu_msg_;
  ros::Publisher predict_pose_odom_pub_;
  geometry_msgs::PoseStamped predict_pose_odom_msg_;
  ros::Publisher predict_pose_imu_odom_pub_;
  geometry_msgs::PoseStamped predict_pose_imu_odom_msg_;
  ros::Publisher ndt_pose_pub_;
  geometry_msgs::PoseStamped ndt_pose_msg_;
  ros::Publisher localizer_pose_pub_;
  geometry_msgs::PoseStamped localizer_pose_msg_;
  geometry_msgs::TwistStamped estimate_twist_msg_;
  ros::Duration scan_duration_;

  double current_velocity_ = 0.0, previous_velocity_ = 0.0, previous_previous_velocity_ = 0.0; // [m/s]
  double current_velocity_x_ = 0.0, previous_velocity_x_ = 0.0;
  double current_velocity_y_ = 0.0, previous_velocity_y_ = 0.0;
  double current_velocity_z_ = 0.0, previous_velocity_z_ = 0.0;
  double current_velocity_imu_x_ = 0.0;
  double current_velocity_imu_y_ = 0.0;
  double current_velocity_imu_z_ = 0.0;
  double current_accel_ = 0.0, previous_accel_ = 0.0; // [m/s^2]
  double current_accel_x_ = 0.0;
  double current_accel_y_ = 0.0;
  double current_accel_z_ = 0.0;
  double angular_velocity_ = 0.0;

  double tf_x_, tf_y_, tf_z_, tf_roll_, tf_pitch_, tf_yaw_;
  Eigen::Matrix4f tf_btol_;
  std::string localizer_ = "velodyne";
  std::string offset_ = "linear"; // linear, zero, quadratic
  bool get_height_ = false;
  bool use_imu_ = false;
  bool use_odom_ = false;
  bool imu_upside_down_ = false;
  bool output_log_data_ = false;
  std::string imu_topic_ = "/imu_raw";
  std::ofstream ofs_;
  std::string filename_;
  sensor_msgs::Imu imu_;
  nav_msgs::Odometry odom_;
  //  tf::TransformListener local_transform_listener;
  tf::StampedTransform local_transform_;

  pthread_mutex_t mutex_;
};

#endif
