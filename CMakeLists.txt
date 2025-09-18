cmake_minimum_required(VERSION 3.16.3)

project(G-KWAY VERSION 1.0)

# set everything up for c++ 17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# test that filesystem header actually is there and works

add_executable(g-kway main/main.cu)
set_target_properties(g-kway PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/exec")
target_link_libraries(g-kway ${PROJECT_SOURCE_DIR}/3rd-party/metis/libmetis.so)
target_link_libraries(g-kway stdc++fs)
#target_link_libraries(g-kway omp ${PROJECT_SOURCE_DIR}/3rd-party/mtmetis/libmtmetis.a)

#------------------------------------------------------------------------------
# default release build
#------------------------------------------------------------------------------

# set compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 --expt-extended-lambda --stack,10485760")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --expt-extended-lambda -arch=compute_86 -code=sm_86")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -Xcompiler -lineinfo --expt-extended-lambda -gencode arch=compute_86,code=sm_86")
#set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -Xcompiler -g -lineinfo --expt-extended-lambda -arch=compute_75 -code=sm_75")
target_compile_options(g-kway PRIVATE -lstdc++fs --extended-lambda)
#if (CMAKE_BUILD_TYPE MATCHES "Debug")
  #target_compile_options(g-kway PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)
#elseif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  #message(STATUS "Setting build type to Release ")
  #set(
    #CMAKE_BUILD_TYPE Release
    #CACHE
    #STRING "Choose the type of build."
    #FORCE
  #)
  ## Set the possible values of build type for cmake-gui
  #set_property(
    #CACHE
    #CMAKE_BUILD_TYPE
    #PROPERTY STRINGS
    #"Debug" "Release"
  #)

#endif()

# build options
option(TF_BUILD_CUDA "Enables builds of cuda code" ON)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)
# Include additional language check
include(CheckLanguage)
# Library-specific variable
set(GKWAY_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)


# -----------------------------------------------------------------------------
# find nvcc
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# -----------------------------------------------------------------------------

if(TF_BUILD_CUDA)
  message(STATUS "Configuring CUDA ...")
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "\nNo CUDA compiler found")
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires CUDA at least v11")
  endif()
endif()  

message("PROJECT_NAME:" ${PROJECT_NAME})
message("PROJECT_SOURCE_DIR:" ${PROJECT_SOURCE_DIR})
message("CMAKE_BUILD_TYPE:" ${CMAKE_BUILD_TYPE})
# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/3rd-party/)
# -----------------------------------------------------------------------------
# include sub cmake list
# -----------------------------------------------------------------------------

# add sub directories
add_subdirectory(main)
#add_subdirectory(unittests)

