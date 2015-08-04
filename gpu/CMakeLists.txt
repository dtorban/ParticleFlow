#
# Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0.
# Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
#
# Code author(s):
# 		Dan Orban (dtorban)
#

cmake_minimum_required (VERSION 2.8.2)
set (CMAKE_VERBOSE_MAKEFILE TRUE)

project (partflow_gpu)

#------------------------------------------
# Library detection
#------------------------------------------

find_package(CUDA)
if (CUDA_FOUND)
        message("CUDA found")
        set(USE_CUDA ON)
        add_definitions(-DUSE_CUDA)
endif()


#------------------------------------------
# Define the source and header files
#------------------------------------------

set (CUDA_SOURCEFILES 
source/math/CudaRandomValue.cu
source/partflow/advectors/CudaVectorFieldAdvector.cu
source/partflow/emitters/CudaEmitterFactory.cu
source/partflow/CudaParticleSet.cu
source/partflow/updaters/CudaParticleUpdater.cu
)

set (CUDA_HEADERFILES 
include/PFGpu/math/CudaRandomValue.cuh
include/PFGpu/partflow/advectors/CudaVectorFieldAdvector.cuh
include/PFGpu/partflow/emitters/CudaEmitterFactory.cuh
include/PFGpu/partflow/emitters/CudaEmitter.cuh
include/PFGpu/partflow/CudaParticleSet.cuh
include/PFGpu/partflow/updaters/CudaParticleUpdater.cuh
)

set (SOURCEFILES 
source/main.cpp
)

set (HEADERFILES 
include/PFGpu/partflow/advectors/GpuVectorFieldAdvector.h
include/PFGpu/partflow/emitters/GpuEmitterFactory.h
include/PFGpu/partflow/GpuParticleUpdater.h
)

if (USE_CUDA)
	set(HEADERFILES ${HEADERFILES} ${CUDA_HEADERFILES})
	set(SOURCEFILES ${SOURCEFILES} ${CUDA_SOURCEFILES})
endif()

include_directories (
  .
  ..
  include
  ../core/include
)

source_group("Header Files" FILES ${HEADERFILES})
source_group("Source Files" FILES ${SOURCEFILES})

#------------------------------------------
# Set output directories to lib, and bin
#------------------------------------------
make_directory(${CMAKE_BINARY_DIR}/lib)
make_directory(${CMAKE_BINARY_DIR}/bin)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach (CONF ${CMAKE_CONFIGURATION_TYPES})
	string (TOUPPER ${CONF} CONF)
	set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} ${CMAKE_BINARY_DIR}/bin)
	set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} ${CMAKE_BINARY_DIR}/lib)
	set (CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} ${CMAKE_BINARY_DIR}/lib)
endforeach(CONF CMAKE_CONFIGURATION_TYPES)

#------------------------------------------
# Build Target
#------------------------------------------

if(USE_CUDA)
	# Append current NVCC flags by something, eg comput capability
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)
	SET( CUDA_PROPAGATE_HOST_FLAGS OFF )
	CUDA_ADD_LIBRARY(${PROJECT_NAME} ${HEADERFILES} ${SOURCEFILES})
	
	if (MSVC)
		# Windows specific code
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} /MP")
	endif (MSVC)
	
else()
	ADD_LIBRARY(${PROJECT_NAME} ${HEADERFILES} ${SOURCEFILES})
endif()

#------------------------------------------
# Install Target
#------------------------------------------

install(DIRECTORY include/ DESTINATION include)

install( TARGETS ${PROJECT_NAME}
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib
         RUNTIME DESTINATION bin)
