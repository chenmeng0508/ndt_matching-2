cmake_minimum_required(VERSION 2.8.3)
project(ndt_matching)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall  -O3 -march=native")
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall  -O0 -march=native")



find_package(PCL REQUIRED)



find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

find_package(CUDA)

if (CUDA_FOUND)
    add_definitions(-DCUDA_FOUND)
    list(APPEND PCL_OPENMP_PACKAGES ndt_gpu)
endif ()

find_package(Eigen3 QUIET)

if (NOT EIGEN3_FOUND)
    # Fallback to cmake_modules
    find_package(cmake_modules REQUIRED)
    find_package(Eigen REQUIRED)
    set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
    set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
    # Possibly map additional variables to the EIGEN3_ prefix.
else ()
    set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif ()




find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        nav_msgs
        tf
        pcl_ros
        sensor_msgs
        pcl_conversions
        velodyne_pointcloud
        ${PCL_OPENMP_PACKAGES}
        )

###################################
## catkin specific configuration ##
###################################
catkin_package(
        CATKIN_DEPENDS std_msgs velodyne_pointcloud  ${PCL_OPENMP_PACKAGES}
        DEPENDS PCL
)




###########
## Build ##
###########
include_directories(${PROJECT_SOURCE_DIR}/src/ndt_cpu/include)
include_directories(${PROJECT_SOURCE_DIR}/src/ndt_tku/include)

include_directories(include ${catkin_INCLUDE_DIRS}
 ${autoware_config_msgs_INCLUDE_DIRS}
 ${autoware_msgs_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")

add_executable(ndt_matching 
src/main.cpp
src/ndt_matching.cpp
src/ndt_cpu/src/NormalDistributionsTransform.cpp
src/ndt_cpu/src/Octree.cpp
src/ndt_cpu/src/Registration.cpp
src/ndt_cpu/src/VoxelGrid.cpp
src/ndt_tku/src/algebra.cpp
src/ndt_tku/src/manage_ND.cpp
src/ndt_tku/src/newton.cpp
)
target_link_libraries(ndt_matching ${catkin_LIBRARIES})
add_dependencies(ndt_matching ${catkin_EXPORTED_TARGETS})


install(TARGETS ndt_matching 
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
